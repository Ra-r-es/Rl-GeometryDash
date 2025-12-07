import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent
from evaluation.evaluate import evaluate_agent


def compare_agents():
    results = {}
    
    # Q-Learning
    if os.path.exists('results/models/q_learning_agent.pkl'):
        env = ImpossibleGameEnv()
        agent = QLearningAgent(env.action_space, env.observation_space)
        agent.load('results/models/q_learning_agent.pkl')
        results['Q-Learning'] = evaluate_agent(agent, env, episodes=100)
        env.close()
    
    # DQN
    if os.path.exists('results/models/dqn_agent.pth'):
        env = ImpossibleGameEnv()
        agent = DQNAgent(env.action_space, env.observation_space)
        agent.load('results/models/dqn_agent.pth')
        results['DQN'] = evaluate_agent(agent, env, episodes=100)
        env.close()
    
    # PPO
    if os.path.exists('results/models/ppo_agent.zip'):
        env = ImpossibleGameEnv()
        agent = PPOAgent(env)
        agent.load('results/models/ppo_agent')
        results['PPO'] = evaluate_agent(agent, env, episodes=100)
        env.close()
    
    # Create comparison table
    df = pd.DataFrame({
        'Agent': list(results.keys()),
        'Mean Score': [r['mean_score'] for r in results.values()],
        'Std Score': [r['std_score'] for r in results.values()],
        'Max Score': [r['max_score'] for r in results.values()],
        'Mean Reward': [r['mean_reward'] for r in results.values()],
        'Mean Length': [r['mean_length'] for r in results.values()]
    })
    
    print("\n=== Comparison Results ===")
    print(df.to_string(index=False))
    
    # Save table
    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/comparison_table.csv', index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mean scores
    agents = list(results.keys())
    mean_scores = [results[a]['mean_score'] for a in agents]
    std_scores = [results[a]['std_score'] for a in agents]
    
    axes[0].bar(agents, mean_scores, yerr=std_scores, capsize=5)
    axes[0].set_title('Mean Score Comparison')
    axes[0].set_ylabel('Score')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Mean rewards
    mean_rewards = [results[a]['mean_reward'] for a in agents]
    axes[1].bar(agents, mean_rewards, color='orange')
    axes[1].set_title('Mean Reward Comparison')
    axes[1].set_ylabel('Reward')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Score distributions (box plot)
    score_data = [results[a]['scores'] for a in agents]
    axes[2].boxplot(score_data, labels=agents)
    axes[2].set_title('Score Distribution')
    axes[2].set_ylabel('Score')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/agents_comparison.png', dpi=300)
    print("\nComparison plot saved to results/plots/agents_comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_agents()
