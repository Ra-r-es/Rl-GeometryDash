"""
Reinforcement Learning Geometry Dash - Main Script

This script orchestrates training, evaluation, and comparison of all RL agents.
Usage:
    python main.py --play                # Play the game manually
    python main.py --train all           # Train all agents
    python main.py --train q_learning    # Train specific agent
    python main.py --evaluate            # Evaluate all trained agents
    python main.py --compare             # Compare agents and generate plots
    python main.py --experiments         # Run hyperparameter experiments
    python main.py --demo <agent>        # Watch trained agent play
    python main.py --full                # Run complete pipeline
"""

import argparse
import os
import sys
from datetime import datetime


def ensure_directories():
    """Create necessary directories."""
    dirs = ['results/models', 'results/logs', 'results/plots', 'results/experiments']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train_agent(agent_type):
    """Train a specific agent."""
    if agent_type == 'q_learning':
        from training.train_q_learning import train_q_learning
        print("\n" + "="*60)
        print("TRAINING Q-LEARNING AGENT")
        print("="*60)
        train_q_learning()

    elif agent_type == 'sarsa':
        from training.train_sarsa import train_sarsa
        print("\n" + "="*60)
        print("TRAINING SARSA AGENT")
        print("="*60)
        train_sarsa()

    elif agent_type == 'dqn':
        from training.train_dqn import train_dqn
        print("\n" + "="*60)
        print("TRAINING DQN AGENT")
        print("="*60)
        train_dqn()

    elif agent_type == 'ppo':
        from training.train_ppo import train_ppo
        print("\n" + "="*60)
        print("TRAINING PPO AGENT")
        print("="*60)
        train_ppo()

    elif agent_type == 'all':
        print("\n" + "="*60)
        print("TRAINING ALL AGENTS")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        train_agent('q_learning')
        train_agent('sarsa')
        train_agent('dqn')
        train_agent('ppo')

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        print(f"Unknown agent type: {agent_type}")
        print("Available: q_learning, sarsa, dqn, ppo, all")


def evaluate_agents():
    """Evaluate all trained agents."""
    from evaluation.evaluate import main as evaluate_main
    print("\n" + "="*60)
    print("EVALUATING ALL AGENTS")
    print("="*60)
    evaluate_main()


def compare_agents():
    """Compare all agents and generate visualizations."""
    from evaluation.compare_agents import compare_agents as compare_main
    print("\n" + "="*60)
    print("COMPARING ALL AGENTS")
    print("="*60)
    compare_main()


def plot_results():
    """Generate training curve plots."""
    from analysis.plot_results import plot_training_curves
    print("\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    plot_training_curves()


def run_experiments():
    """Run hyperparameter tuning experiments."""
    from experiments.hyperparameter_tuning import run_all_experiments
    print("\n" + "="*60)
    print("RUNNING HYPERPARAMETER EXPERIMENTS")
    print("="*60)
    run_all_experiments()


def demo_agent(agent_type):
    """Watch a trained agent play the game."""
    from environment import ImpossibleGameEnv

    print(f"\n" + "="*60)
    print(f"DEMO: {agent_type.upper()} AGENT")
    print("="*60)

    env = ImpossibleGameEnv(render_mode="human", max_steps=10000)

    if agent_type == 'q_learning':
        from agents.tabular.q_learning_agent import QLearningAgent
        agent = QLearningAgent(env.action_space, env.observation_space)
        model_path = 'results/models/q_learning_agent.pkl'

    elif agent_type == 'sarsa':
        from agents.tabular.sarsa_agent import SARSAAgent
        agent = SARSAAgent(env.action_space, env.observation_space)
        model_path = 'results/models/sarsa_agent.pkl'

    elif agent_type == 'dqn':
        from agents.deep.dqn_agent import DQNAgent
        agent = DQNAgent(env.action_space, env.observation_space)
        model_path = 'results/models/dqn_agent.pth'

    elif agent_type == 'ppo':
        from agents.policy.ppo_agent import PPOAgent
        agent = PPOAgent(env)
        model_path = 'results/models/ppo_agent.zip'

    else:
        print(f"Unknown agent: {agent_type}")
        env.close()
        return

    if not os.path.exists(model_path.replace('.zip', '.zip') if '.zip' in model_path else model_path):
        print(f"Model not found: {model_path}")
        print("Train the agent first!")
        env.close()
        return

    agent.load(model_path.replace('.zip', '') if agent_type == 'ppo' else model_path)
    print(f"Loaded model from: {model_path}")
    print("Press Ctrl+C to exit\n")

    try:
        episodes = 0
        while True:
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                env.render()

            episodes += 1
            print(f"Episode {episodes}: Score = {info['score']}, Reward = {total_reward:.0f}")

    except KeyboardInterrupt:
        print("\nDemo ended.")

    env.close()


def play_game():
    """Play the game manually with keyboard controls."""
    from play_game import play_game as play
    play()


def run_full_pipeline():
    """Run the complete training and evaluation pipeline."""
    print("\n" + "="*70)
    print("FULL PIPELINE: RL Geometry Dash")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Train all agents
    train_agent('all')

    # Generate training plots
    plot_results()

    # Evaluate all agents
    evaluate_agents()

    # Compare agents
    compare_agents()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - results/models/          (trained models)")
    print("  - results/logs/            (training metrics)")
    print("  - results/plots/           (visualizations)")
    print("  - results/comparison_table.csv")
    print("  - results/detailed_comparison.csv")


def main():
    parser = argparse.ArgumentParser(
        description='RL Geometry Dash - Train and evaluate reinforcement learning agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --play                Play the game manually (SPACE to jump)
  python main.py --train all           Train all agents
  python main.py --train dqn           Train only DQN agent
  python main.py --evaluate            Evaluate all trained agents
  python main.py --compare             Compare agents with visualizations
  python main.py --plots               Generate training curve plots
  python main.py --experiments         Run hyperparameter tuning
  python main.py --demo ppo            Watch PPO agent play
  python main.py --full                Run complete pipeline

Available agents: q_learning, sarsa, dqn, ppo, all
        """
    )

    parser.add_argument('--play', action='store_true',
                        help='Play the game manually with keyboard (SPACE to jump)')
    parser.add_argument('--train', type=str, metavar='AGENT',
                        help='Train agent(s): q_learning, sarsa, dqn, ppo, or all')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate all trained agents')
    parser.add_argument('--compare', action='store_true',
                        help='Compare agents and generate comparison plots')
    parser.add_argument('--plots', action='store_true',
                        help='Generate training curve plots')
    parser.add_argument('--experiments', action='store_true',
                        help='Run hyperparameter tuning experiments')
    parser.add_argument('--demo', type=str, metavar='AGENT',
                        help='Demo a trained agent: q_learning, sarsa, dqn, ppo')
    parser.add_argument('--full', action='store_true',
                        help='Run full pipeline: train all, evaluate, compare')

    args = parser.parse_args()

    # Check if any argument was provided
    if not any(vars(args).values()):
        parser.print_help()
        print("\n" + "-"*60)
        print("Quick start: python main.py --full")
        print("-"*60)
        return

    ensure_directories()

    if args.play:
        play_game()
        return

    if args.train:
        train_agent(args.train.lower())

    if args.evaluate:
        evaluate_agents()

    if args.compare:
        compare_agents()

    if args.plots:
        plot_results()

    if args.experiments:
        run_experiments()

    if args.demo:
        demo_agent(args.demo.lower())

    if args.full:
        run_full_pipeline()


if __name__ == "__main__":
    main()
