import numpy as np
from collections import deque
from magent2.environments import battle_v4
import torch

def evaluate_checkpoint_noise(q_network, num_episodes=5):
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

                if red_agents:
                    states_red = torch.stack([
                        torch.tensor(observations[agent], dtype=torch.float32)
                        for agent in red_agents
                    ]).to(device)
                    with torch.no_grad():
                        q_network.reset_noise()  # Reset noise before action selection
                        q_values_red = q_network(states_red)
                        selected_actions_red = torch.argmax(q_values_red, dim=1)

                    for idx, agent in enumerate(red_agents):
                        actions[agent] = selected_actions_red[idx].item()
                
                if blue_agents:
                    for idx, agent in enumerate(blue_agents):
                        actions[agent] = env.action_spaces[agent].sample()

                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                done = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in env.agents}
                done_agents.update([agent for agent, finished in done.items() if finished])
                observations = next_obs
                steps += 1

                episode_reward += sum(rewards.get(agent, 0.0) for agent in red_agents)

                if all(done.values()):
                    total_steps += steps
                    total_rewards += episode_reward
                    break
        return total_steps / num_episodes, total_rewards / num_episodes
