import torch
from magent2.environments import battle_v4
# --- Function to evaluate a checkpoint ---
def evaluate_checkpoint(q_network, num_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = battle_v4.parallel_env(map_size=45, max_cycles=1000, minimap_mode=False) 
    total_steps = 0
    total_rewards = 0
    for _ in range(num_episodes):
        observations = env.reset()
        done_agents = set()
        steps = 0
        episode_reward = 0
        while True:
            actions = {}
            red_agents = [agent for agent in env.agents if agent.startswith("red_") and agent not in done_agents]
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_") and agent not in done_agents]

            if blue_agents:
                states_blue = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in blue_agents
                ]).to(device)
                with torch.no_grad():
                    q_network.reset_noise() # Reset noise before action selection
                    q_values_blue = q_network(states_blue)
                    selected_actions_blue = torch.argmax(q_values_blue, dim=1)

                for idx, agent in enumerate(blue_agents):
                    actions[agent] = selected_actions_blue[idx].item()
            
            if red_agents:
                for idx, agent in enumerate(red_agents):
                  actions[agent] = env.action_spaces[agent].sample()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update done_agents based on terminations
            done_agents.update([agent for agent, terminated in terminations.items() if terminated])

            observations = next_obs
            steps += 1

            episode_reward += sum(rewards.get(agent, 0.0) for agent in blue_agents)

            # Check if all agents are done (either terminated or truncated)
            if all(terminations.values()) or all(truncations.values()):
                total_steps += steps
                total_rewards += episode_reward
                break
    return total_steps / num_episodes, total_rewards / num_episodes
