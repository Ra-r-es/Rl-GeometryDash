import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.policy.ppo_agent import PPOAgent
from training.config import PPO_CONFIG

def train_ppo(config=PPO_CONFIG):
    """Train PPO agent using Stable-Baselines3."""
    
    env = ImpossibleGameEnv(max_steps=10000)
    
    agent = PPOAgent(
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef']
    )
    
    print("Training PPO agent...")
    agent.update(total_timesteps=config['total_timesteps'])
    
    os.makedirs('results/models', exist_ok=True)
    agent.save('results/models/ppo_agent')
    
    print("PPO training complete!")


if __name__ == "__main__":
    train_ppo()
