import gymnasium as gym
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent


def visualize_agent(agent_type, model_path, episodes=5):
    """Vizualizează un agent în acțiune."""
    
    env = ImpossibleGameEnv(render_mode="human")
    
    # Load agent
    if agent_type == "q_learning":
        agent = QLearningAgent(env.action_space, env.observation_space)
        agent.load(model_path)
    elif agent_type == "dqn":
        agent = DQNAgent(env.action_space, env.observation_space)
        agent.load(model_path)
    elif agent_type == "ppo":
        agent = PPOAgent(env)
        agent.load(model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print(f"Visualizing {agent_type} agent...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            env.render()
        
        print(f"Episode {episode+1}: Score = {info['score']}, Reward = {total_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained agent')
    parser.add_argument('--agent', type=str, required=True, 
                    choices=['q_learning', 'dqn', 'ppo'],
                    help='Agent type')
    parser.add_argument('--model', type=str, required=True,
                    help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                    help='Number of episodes to visualize')
    
    args = parser.parse_args()
    
    visualize_agent(args.agent, args.model, args.episodes)
