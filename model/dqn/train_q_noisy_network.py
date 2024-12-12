import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from noisy_linear import DuelingNoisyQNetwork
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from magent2.environments import battle_v4
from evaluate.evaluate_checkpoint_noise import evaluate_checkpoint_noise
import os

def train():
    # --- Initialize environment and device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard writer ---
    log_dir = os.path.join("runs", "Dueling_Noisy_DDQN_BattleV4_PER_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    # --- Initialize the environment ---
    env = battle_v4.parallel_env(map_size=45, max_cycles=1000, minimap_mode=False)

    sample_observation = env.observation_spaces[env.agents[0]].shape
    state_space = sample_observation
    action_space = env.action_spaces[env.agents[0]].n

    # --- Initialize networks and optimizers ---
    q_network_blue = DuelingNoisyQNetwork(state_space, action_space).to(device)
    target_network_blue = DuelingNoisyQNetwork(state_space, action_space).to(device)
    target_network_blue.load_state_dict(q_network_blue.state_dict())
    target_network_blue.eval()

    optimizer_blue = optim.Adam(q_network_blue.parameters(), lr=0.0005)

    # --- Hyperparameters ---
    gamma = 0.99
    num_episodes = 250
    max_steps_per_episode = 1000
    checkpoint_interval = 5
    polyak_tau = 0.005
    replay_buffer_size = 500000
    batch_size = 1024
    num_val_episodes = 5

    # --- Initialize prioritized replay buffers ---
    replay_buffer_blue = PrioritizedReplayBuffer(replay_buffer_size)
    # --- Training loop ---
    best_val = float('inf')
    best_checkpoint = ""
    for episode in range(0, num_episodes + 1):
        observations = env.reset()
        total_reward_blue = 0
        done_agents = set()

        for step in range(max_steps_per_episode):
            actions = {}
            red_agents = [agent for agent in env.agents if agent.startswith("red_") and agent not in done_agents]
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_") and agent not in done_agents]

            # --- Process all blue agents ---
            if blue_agents:
                states_blue = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in blue_agents
                ]).to(device)

                # --- Select actions with Noisy Networks ---
                q_network_blue.reset_noise()
                with torch.no_grad():
                    q_values_blue = q_network_blue(states_blue)
                    selected_actions_blue = torch.argmax(q_values_blue, dim=1)

                for idx, agent in enumerate(blue_agents):
                    actions[agent] = selected_actions_blue[idx].item()

            # --- Process all red agents ---
            if red_agents:
                selected_actions_red = torch.randint(0, action_space, (len(red_agents),), device=device)
                for idx, agent in enumerate(red_agents):
                    actions[agent] = selected_actions_red[idx].item()

            # --- Take step in environment ---
            
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            # --- Update total rewards ---
            total_reward_blue += sum(rewards.get(agent, 0.0) for agent in blue_agents)
            # --- Store experiences in replay buffers ---
            for agent in blue_agents:
                next_state = next_obs.get(agent) if not terminations.get(agent, False) else observations[agent]  # Use current state as next state if agent is done
                replay_buffer_blue.push(
                    observations[agent],
                    actions[agent],
                    rewards.get(agent, 0.0),
                    next_state,
                    terminations.get(agent, False)  # Only consider termination as done
                )
            # --- Train blue network on a batch sampled from replay buffer ---
            if len(replay_buffer_blue) >= batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = replay_buffer_blue.sample(batch_size)

                current_states_blue = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
                current_actions_blue = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(device)
                next_states_blue = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
                rewards_blue = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                dones_blue = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(device)
                weights_tensor_blue = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

                q_network_blue.reset_noise()
                target_network_blue.reset_noise()
                current_q_values_blue = q_network_blue(current_states_blue).gather(1, current_actions_blue)
                with torch.no_grad():
                    actions = q_network_blue(next_states_blue).max(1, keepdim=True)[1]
                    next_q_values_blue = target_network_blue(next_states_blue).gather(1, actions)
                    target_q_values_blue = rewards_blue + gamma * next_q_values_blue * (1 - dones_blue)

                td_errors = (target_q_values_blue - current_q_values_blue).abs().detach().cpu().numpy()
                new_priorities = td_errors + 1e-6
                replay_buffer_blue.update_priorities(indices, new_priorities)

                loss_blue = (weights_tensor_blue * nn.SmoothL1Loss(reduction='none')(current_q_values_blue, target_q_values_blue)).mean()

                optimizer_blue.zero_grad()
                loss_blue.backward()
                nn.utils.clip_grad_norm_(q_network_blue.parameters(), 1.0)
                optimizer_blue.step()

                writer.add_scalar('Loss/Blue', loss_blue.item(), episode * max_steps_per_episode + step)

            # --- Soft update target networks ---
            for target_param, param in zip(target_network_blue.parameters(), q_network_blue.parameters()):
                target_param.data.copy_(polyak_tau * param.data + (1 - polyak_tau) * target_param.data)

            observations = next_obs
            done_agents.update([agent for agent, terminated in terminations.items() if terminated])

            if all(terminations.values()) or (truncations and all(truncations.values())):
                break

        # --- Log episode statistics ---
        writer.add_scalar('Total Reward/Blue', total_reward_blue, episode)
        print(f"Episode {episode}/{num_episodes}, Total Reward Blue: {total_reward_blue:.2f}")

        # --- Save checkpoints and evaluate ---
        if episode % checkpoint_interval == 0:
            checkpoint_path = f"blue_agent_dueling_noisy_ddqn_per_ep{episode}.pth"
            torch.save(q_network_blue.state_dict(), checkpoint_path)

            val, val_reward = evaluate_checkpoint(q_network_blue, num_val_episodes)
            writer.add_scalar('Validation/Average Steps', val, episode)
            writer.add_scalar('Validation/Average Reward', val_reward, episode)
            print(f"Episode {episode}: Validation average steps = {val:.2f}, Validation average reward = {val_reward:.2f}")

            if val <= best_val:
                best_val = val
                if best_checkpoint:
                    os.remove(best_checkpoint)
                print(f"New best checkpoint saved at {best_checkpoint} with average steps {best_val:.2f}")
            if val > best_val:
                os.remove(checkpoint_path)
    if best_checkpoint:
        best_checkpoint_final = "blue_agent_dueling_noisy_ddqn_per_best.pth"
        os.rename(best_checkpoint, best_checkpoint_final)
        print(f"Training complete. Best model saved at {best_checkpoint_final}")
    else:
        torch.save(q_network_blue.state_dict(), "blue_agent_dueling_noisy_ddqn_per_final.pth")
        print(f"Training complete. Final model saved at blue_agent_dueling_noisy_ddqn_per_final.pth")