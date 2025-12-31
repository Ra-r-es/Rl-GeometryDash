"""
Training configurations for all RL agents.
"""

# Q-Learning config (off-policy value-based)
Q_LEARNING_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 10,
    'episodes': 5000,
    'max_steps': 10000
}

# SARSA config (on-policy value-based)
SARSA_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 10,
    'episodes': 5000,
    'max_steps': 10000
}

# DQN config (deep value-based with experience replay)
DQN_CONFIG = {
    'learning_rate': 1e-4,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'buffer_size': 100000,
    'batch_size': 64,
    'target_update_freq': 1000,
    'episodes': 2000,
    'max_steps': 10000
}

# PPO config (policy gradient with clipped objective)
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'total_timesteps': 1000000
}

# Hyperparameter experiment configs
EXPERIMENT_CONFIGS = {
    'learning_rates': [0.01, 0.05, 0.1, 0.2],
    'discount_factors': [0.9, 0.95, 0.99],
    'epsilon_decays': [0.999, 0.9995, 0.9999],
    'dqn_batch_sizes': [32, 64, 128],
    'dqn_learning_rates': [1e-5, 1e-4, 1e-3],
}