from stable_baselines3 import PPO
from agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """Wrapper pentru Stable-Baselines3 PPO."""
    def __init__(self, env, **kwargs):
        super().__init__(env.action_space, env.observation_space)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            **kwargs
        )
    
    def select_action(self, observation, training=True):
        """Predict action."""
        action, _ = self.model.predict(observation, deterministic=not training)
        return action
    
    def update(self, total_timesteps):
        """Train model."""
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path):
        """Save model."""
        self.model.save(path)
        print(f"PPO agent saved to {path}")
    
    def load(self, path):
        """Load model."""
        self.model = PPO.load(path)
        print(f"PPO agent loaded from {path}")