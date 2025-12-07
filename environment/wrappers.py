import gymnasium as gym
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
    """
    Agentul ia decizie la fiecare `skip` frames.
    Reward-urile se acumuleaza.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalizează observațiile pentru a fi în [-1, 1].
    """
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return np.clip(obs, -1, 1)


class RewardShapingWrapper(gym.Wrapper):
    """
    Modifică reward-urile pentru a ajuta invatarea.
    """
    def __init__(self, env, distance_reward_scale=0.01):
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.last_distance = 0
    
    def reset(self, **kwargs):
        self.last_distance = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Bonus pentru distanță parcursă
        distance_reward = (info['distance'] - self.last_distance) * self.distance_reward_scale
        self.last_distance = info['distance']
        
        reward += distance_reward
        
        return obs, reward, terminated, truncated, info