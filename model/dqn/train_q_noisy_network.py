import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from noisy_linear import DuelingNoisyQNetwork
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from magent2.environments import battle_v4
from evaluate_checkpoint_noise import evaluate_checkpoint_noise
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join("runs", "Dueling_Noisy_DDQN_BattleV4_PER_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)


    env = battle_v4.parallel_env(map_size=45, max_cycles=1000, minimap_mode=False)

    sample_observation = env.observation_spaces[env.agents[0]].shape
    state_space = sample_observation
    action_space = env.action_spaces[env.agents[0]].n

    # --- Initialize networks and optimizers ---
    q_network_red = DuelingNoisyQNetwork(state_space, action_space).to(device)
    target_network_red = DuelingNoisyQNetwork(state_space, action_space).to(device)
    target_network_red.load_state_dict(q_network_red.state_dict())
    target_network_red.eval()

    optimizer_red = optim.Adam(q_network_red.parameters(), lr=0.0005)

    gamma = 0.99
    num_episodes = 150
    max_steps_per_episode = 1000
    checkpoint_interval = 5
    polyak_tau = 0.005
    replay_buffer_size = 800000
    batch_size = 128
    num_val_episodes = 5

    # --- Initialize prioritized replay buffers ---
    replay_buffer_red = PrioritizedReplayBuffer(replay_buffer_size)
    # --- Training loop ---
    best_val = float('inf')
    best_checkpoint = ""
    for episode in range(0, num_episodes + 1):
        observations = env.reset()
        total_reward_red = 0
        done_agents = set()

        for step in range(max_steps_per_episode):
            actions = {}
            red_agents = [agent for agent in env.agents if agent.startswith("red_") and agent not in done_agents]
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_") and agent not in done_agents]

            # --- Process all red agents ---
            if red_agents:
                states_red = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in red_agents
                ]).to(device)

                # --- Select actions with Noisy Networks ---
                q_network_red.reset_noise()
                with torch.no_grad():
                    q_values_red = q_network_red(states_red)
                    selected_actions_red = torch.argmax(q_values_red, dim=1)

                for idx, agent in enumerate(red_agents):
                    actions[agent] = selected_actions_red[idx].item()

            # --- Process all blue agents ---
            if blue_agents:
                selected_actions_blue = torch.randint(0, action_space, (len(blue_agents),), device=device)
                for idx, agent in enumerate(blue_agents):
                    actions[agent] = selected_actions_blue[idx].item()

            # --- Take step in environment ---
            
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in env.agents}
            # --- Update total rewards ---
            total_reward_red += sum(rewards.get(agent, 0.0) for agent in red_agents)
            # --- Store experiences in replay buffers ---
            for agent in red_agents:
                if agent in done:  # Check if agent is still active
                    replay_buffer_red.push(
                        observations[agent],
                        actions[agent],
                        rewards.get(agent, 0.0),
                        next_obs[agent],
                        done[agent]
                    )
            # --- Train red network on a batch sampled from replay buffer ---
            if len(replay_buffer_red) >= batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = replay_buffer_red.sample(batch_size)

                current_states_red = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
                current_actions_red = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(device)
                next_states_red = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
                rewards_red = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                dones_red = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(device)
                weights_tensor_red = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

                q_network_red.reset_noise()
                target_network_red.reset_noise()
                current_q_values_red = q_network_red(current_states_red).gather(1, current_actions_red)
                with torch.no_grad():
                    actions = q_network_red(next_states_red).max(1, keepdim=True)[1]
                    next_q_values_red = target_network_red(next_states_red).gather(1, actions)
                    target_q_values_red = rewards_red + gamma * next_q_values_red * (1 - dones_red)

                td_errors = (target_q_values_red - current_q_values_red).abs().detach().cpu().numpy()
                new_priorities = td_errors + 1e-6
                replay_buffer_red.update_priorities(indices, new_priorities)

                loss_red = (weights_tensor_red * nn.SmoothL1Loss(reduction='none')(current_q_values_red, target_q_values_red)).mean()

                optimizer_red.zero_grad()
                loss_red.backward()
                nn.utils.clip_grad_norm_(q_network_red.parameters(), 1.0)
                optimizer_red.step()

                writer.add_scalar('Loss/Red', loss_red.item(), episode * max_steps_per_episode + step)

            # --- Soft update target networks ---
            for target_param, param in zip(target_network_red.parameters(), q_network_red.parameters()):
                target_param.data.copy_(polyak_tau * param.data + (1 - polyak_tau) * target_param.data)

            observations = next_obs
            done_agents.update([agent for agent, finished in done.items() if finished])

            if all(done.values()):
                break

        # --- Log episode statistics ---
        writer.add_scalar('Total Reward/Red', total_reward_red, episode)
        print(f"Episode {episode}/{num_episodes}, Total Reward Red: {total_reward_red:.2f}")

        # --- Save checkpoints and evaluate ---
        if episode % checkpoint_interval == 0:
            checkpoint_path = f"red_agent_dueling_noisy_ddqn_per_ep{episode}.pth"
            torch.save(q_network_red.state_dict(), checkpoint_path)

            val, val_reward = evaluate_checkpoint_noise(q_network_red, num_val_episodes)
            writer.add_scalar('Validation/Average Steps', val, episode)
            writer.add_scalar('Validation/Average Reward', val_reward, episode)
            print(f"Episode {episode}: Validation average steps = {val:.2f}, Validation average reward = {val_reward:.2f}")

            if val < best_val:
                best_val = val
                if best_checkpoint:
                    os.remove(best_checkpoint)
                best_checkpoint = checkpoint_path
                print(f"New best checkpoint saved at {best_checkpoint} with average steps {best_val:.2f}")

    # --- Save final model (best checkpoint) ---
    if best_checkpoint:
        print(f"Training complete. Best model saved at {best_checkpoint}")
    else:
        torch.save(q_network_red.state_dict(), "red_agent_dueling_noisy_ddqn_per_final.pth")
        print(f"Training complete. Best model saved at red_agent_dueling_noisy_ddqn_per_final.pth")