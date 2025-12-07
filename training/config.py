"""
Configurări comune pentru training.
"""

# Q-Learning config
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

# DQN config
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

# PPO config
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