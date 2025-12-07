import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent


def evaluate_agent(agent, env, episodes=100, render=False):
    """Evaluează un agent pe N episoade."""
    
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}: Score = {info['score']}, Reward = {episode_reward:.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_score': np.mean(episode_scores),
        'std_score': np.std(episode_scores),
        'mean_length': np.mean(episode_lengths),
        'max_score': np.max(episode_scores),
        'rewards': episode_rewards,
        'scores': episode_scores
    }


def main():
    # Evaluare Q-Learning
    print("\n=== Evaluating Q-Learning ===")
    env = ImpossibleGameEnv()
    agent_q = QLearningAgent(env.action_space, env.observation_space)
    
    if os.path.exists('results/models/q_learning_agent.pkl'):
        agent_q.load('results/models/q_learning_agent.pkl')
        results_q = evaluate_agent(agent_q, env, episodes=100)
        print(f"Q-Learning - Mean Score: {results_q['mean_score']:.2f} ± {results_q['std_score']:.2f}")
    else:
        print("Q-Learning model not found!")
    
    env.close()
    
    # Evaluare DQN
    print("\n=== Evaluating DQN ===")
    env = ImpossibleGameEnv()
    agent_dqn = DQNAgent(env.action_space, env.observation_space)
    
    if os.path.exists('results/models/dqn_agent.pth'):
        agent_dqn.load('results/models/dqn_agent.pth')
        results_dqn = evaluate_agent(agent_dqn, env, episodes=100)
        print(f"DQN - Mean Score: {results_dqn['mean_score']:.2f} ± {results_dqn['std_score']:.2f}")
    else:
        print("DQN model not found!")
    
    env.close()
    
    # Evaluare PPO
    print("\n=== Evaluating PPO ===")
    env = ImpossibleGameEnv()
    
    if os.path.exists('results/models/ppo_agent.zip'):
        agent_ppo = PPOAgent(env)
        agent_ppo.load('results/models/ppo_agent')
        results_ppo = evaluate_agent(agent_ppo, env, episodes=100)
        print(f"PPO - Mean Score: {results_ppo['mean_score']:.2f} ± {results_ppo['std_score']:.2f}")
    else:
        print("PPO model not found!")
    
    env.close()


if __name__ == "__main__":
    main()
