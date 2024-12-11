import torch
from magent2.environments import battle_v4

def evaluate_checkpoint(q_network, device, num_episodes=5):
    """
    Evaluates a trained Q-network on the Battle environment.

    Parameters:
    - q_network: The trained Q-network to evaluate.
    - device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').
    - num_episodes: Number of episodes to run for evaluation.

    Returns:
    - average_steps: Average number of steps taken across episodes.
    - average_rewards: Average total rewards achieved across episodes.
    """
    # Initialize the environment
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

            # Select actions for red agents using the Q-network
            if red_agents:
                states_red = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in red_agents
                ]).to(device)

                with torch.no_grad():
                    q_values_red = q_network(states_red)
                    selected_actions_red = torch.argmax(q_values_red, dim=1)

                for idx, agent in enumerate(red_agents):
                    actions[agent] = selected_actions_red[idx].item()

            # Select random actions for blue agents
            if blue_agents:
                for agent in blue_agents:
                    actions[agent] = env.action_spaces[agent].sample()

            # Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update done agents
            done_agents.update([agent for agent, terminated in terminations.items() if terminated])

            # Accumulate rewards for red agents
            episode_reward += sum(rewards.get(agent, 0.0) for agent in red_agents)
            observations = next_obs
            steps += 1

            # Check if all agents are done (either terminated or truncated)
            if all(terminations.values()) or all(truncations.values()):
                total_steps += steps
                total_rewards += episode_reward
                break

    # Compute averages
    average_steps = total_steps / num_episodes
    average_rewards = total_rewards / num_episodes
    return average_steps, average_rewards
