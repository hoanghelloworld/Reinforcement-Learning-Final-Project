import torch
from magent2.environments import battle_v4
def evaluate_checkpoint(q_network, red_q_network, pretrain_policy, num_episodes=5):
    env = battle_v4.parallel_env(map_size=45, max_cycles=1000, minimap_mode=False)
    total_steps = 0
    total_rewards_random = 0
    total_rewards_pretrained = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for _ in range(num_episodes):
        # --- Evaluate against random policy ---
        observations = env.reset()
        done_agents = set()
        episode_reward_random = 0
        while True:
            actions = {}
            red_agents = [agent for agent in env.agents if agent.startswith("red_") and agent not in done_agents]
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_") and agent not in done_agents]

            if red_agents:
                for agent in red_agents:
                    actions[agent] = env.action_spaces[agent].sample()  # Random action

            if blue_agents:
                states_blue = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in blue_agents
                ]).to(device)
                with torch.no_grad():
                    q_network_blue.reset_noise()          
                    q_values_blue = q_network(states_blue)
                    selected_actions_blue = torch.argmax(q_values_blue, dim=1)

                for idx, agent in enumerate(blue_agents):
                    actions[agent] = selected_actions_blue[idx].item()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done_agents.update([agent for agent, terminated in terminations.items() if terminated])
            observations = next_obs
            episode_reward_random += sum(rewards.get(agent, 0.0) for agent in blue_agents)
            if all(terminations.values()) or all(truncations.values()):
                total_rewards_random += episode_reward_random
                break

        # --- Evaluate against pretrained policy ---
        observations = env.reset()
        done_agents = set()
        episode_reward_pretrained = 0
        while True:
            actions = {}
            red_agents = [agent for agent in env.agents if agent.startswith("red_") and agent not in done_agents]
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_") and agent not in done_agents]

            if red_agents:
                for agent in red_agents:
                    actions[agent] = pretrain_policy(env, agent, observations[agent])  # Pretrained action

            if blue_agents:
                states_blue = torch.stack([
                    torch.tensor(observations[agent], dtype=torch.float32)
                    for agent in blue_agents
                ]).to(device)
                with torch.no_grad():
                    q_network_blue.reset_noise() 
                    q_values_blue = q_network(states_blue)
                    selected_actions_blue = torch.argmax(q_values_blue, dim=1)

                for idx, agent in enumerate(blue_agents):
                    actions[agent] = selected_actions_blue[idx].item()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done_agents.update([agent for agent, terminated in terminations.items() if terminated])
            observations = next_obs
            episode_reward_pretrained += sum(rewards.get(agent, 0.0) for agent in blue_agents)
            if all(terminations.values()) or all(truncations.values()):
                total_rewards_pretrained += episode_reward_pretrained
                break

    avg_reward_random = total_rewards_random / num_episodes
    avg_reward_pretrained = total_rewards_pretrained / num_episodes
    avg_reward = (avg_reward_random + avg_reward_pretrained) / 2

    return avg_reward, avg_reward_random, avg_reward_pretrained