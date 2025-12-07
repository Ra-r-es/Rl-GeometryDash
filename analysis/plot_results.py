import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("darkgrid")


def plot_training_curves():
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Q-Learning
    if os.path.exists('results/logs/q_learning_rewards.npy'):
        q_rewards = np.load('results/logs/q_learning_rewards.npy')
        q_scores = np.load('results/logs/q_learning_scores.npy')
        
        # Smooth curves
        window = 50
        q_rewards_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        q_scores_smooth = np.convolve(q_scores, np.ones(window)/window, mode='valid')
        
        axes[0, 0].plot(q_rewards, alpha=0.3, color='blue')
        axes[0, 0].plot(q_rewards_smooth, color='blue', linewidth=2)
        axes[0, 0].set_title('Q-Learning: Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        axes[1, 0].plot(q_scores, alpha=0.3, color='green')
        axes[1, 0].plot(q_scores_smooth, color='green', linewidth=2)
        axes[1, 0].set_title('Q-Learning: Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
    
    # DQN
    if os.path.exists('results/logs/dqn_rewards.npy'):
        dqn_rewards = np.load('results/logs/dqn_rewards.npy')
        dqn_scores = np.load('results/logs/dqn_scores.npy')
        
        window = 50
        dqn_rewards_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        dqn_scores_smooth = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
        
        axes[0, 1].plot(dqn_rewards, alpha=0.3, color='red')
        axes[0, 1].plot(dqn_rewards_smooth, color='red', linewidth=2)
        axes[0, 1].set_title('DQN: Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        
        axes[1, 1].plot(dqn_scores, alpha=0.3, color='orange')
        axes[1, 1].plot(dqn_scores_smooth, color='orange', linewidth=2)
        axes[1, 1].set_title('DQN: Scores')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score')
    
    # Comparison (if both exist)
    if os.path.exists('results/logs/q_learning_scores.npy') and \
       os.path.exists('results/logs/dqn_scores.npy'):
        
        axes[0, 2].plot(q_scores_smooth, label='Q-Learning', color='green', linewidth=2)
        axes[0, 2].plot(dqn_scores_smooth, label='DQN', color='orange', linewidth=2)
        axes[0, 2].set_title('Score Comparison')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        
        axes[1, 2].hist(q_scores, bins=30, alpha=0.5, label='Q-Learning', color='green')
        axes[1, 2].hist(dqn_scores, bins=30, alpha=0.5, label='DQN', color='orange')
        axes[1, 2].set_title('Score Distribution')
        axes[1, 2].set_xlabel('Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/training_curves.png', dpi=300)
    print("Training curves saved to results/plots/training_curves.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)
    plot_training_curves()
