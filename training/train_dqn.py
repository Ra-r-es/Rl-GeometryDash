import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.deep.dqn_agent import DQNAgent
from training.config import DQN_CONFIG


def train_dqn(config=DQN_CONFIG, render=False):
    """Train DQN agent."""
    
    render_mode = "human" if render else None
    env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])
    
    agent = DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    
    episode_rewards = []
    episode_scores = []
    
    for episode in tqdm(range(config['episodes']), desc="Training DQN"):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_score = np.mean(episode_scores[-50:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Score = {avg_score:.0f}, Epsilon = {agent.epsilon:.3f}")
    
    env.close()
    
    os.makedirs('results/models', exist_ok=True)
    agent.save('results/models/dqn_agent.pth')
    
    np.save('results/logs/dqn_rewards.npy', episode_rewards)
    np.save('results/logs/dqn_scores.npy', episode_scores)
    
    return episode_rewards, episode_scores


if __name__ == "__main__":
    rewards, scores = train_dqn(render=False)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('DQN: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(scores)
    plt.title('DQN: Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('results/plots/dqn_training.png')
    plt.show()