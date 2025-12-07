import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
BLUE = (100, 150, 255)
YELLOW = (255, 220, 0)
GRAY = (100, 100, 100)

# Game settings
PLAYER_SIZE = 30
GRAVITY = 0.8
JUMP_FORCE = -15
SCROLL_SPEED = 6
GROUND_HEIGHT = HEIGHT - 100

class Player:
    def __init__(self):
        self.x = 150
        self.y = GROUND_HEIGHT - PLAYER_SIZE
        self.vel_y = 0
        self.on_ground = True
        self.dead = False
        self.rotation = 0
        
    def jump(self):
        if self.on_ground and not self.dead:
            self.vel_y = JUMP_FORCE
            self.on_ground = False
    
    def update(self):
        if self.dead:
            return
            
        self.vel_y += GRAVITY
        self.y += self.vel_y
        
        if self.y >= GROUND_HEIGHT - PLAYER_SIZE:
            self.y = GROUND_HEIGHT - PLAYER_SIZE
            self.vel_y = 0
            self.on_ground = True
            
        if not self.on_ground:
            self.rotation += 10
        else:
            self.rotation = 0
    
    def draw(self, screen):
        surf = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(surf, BLUE, (0, 0, PLAYER_SIZE, PLAYER_SIZE))
        rotated = pygame.transform.rotate(surf, self.rotation)
        rect = rotated.get_rect(center=(self.x + PLAYER_SIZE//2, self.y + PLAYER_SIZE//2))
        screen.blit(rotated, rect)
    
    def get_rect(self):
        return pygame.Rect(self.x + 2, self.y + 2, PLAYER_SIZE - 4, PLAYER_SIZE - 4)

class Obstacle:
    def __init__(self, x, obs_type):
        self.x = x
        self.type = obs_type
        
        if obs_type == "spike":
            self.width = 30
            self.height = 40
            self.y = GROUND_HEIGHT - self.height
        elif obs_type == "gap":
            self.width = 100
            self.height = 50
            self.y = GROUND_HEIGHT
        elif obs_type == "platform":
            self.width = 80
            self.height = 20
            self.y = GROUND_HEIGHT - 120
    
    def draw(self, screen):
        if self.type == "spike":
            points = [
                (self.x, self.y + self.height),
                (self.x + self.width//2, self.y),
                (self.x + self.width, self.y + self.height)
            ]
            pygame.draw.polygon(screen, RED, points)
        elif self.type == "gap":
            pass
        elif self.type == "platform":
            pygame.draw.rect(screen, YELLOW, (self.x, self.y, self.width, self.height))
    
    def collides_with(self, player):
        player_rect = player.get_rect()
        
        if self.type == "spike":
            spike_rect = pygame.Rect(self.x + 5, self.y, self.width - 10, self.height)
            return player_rect.colliderect(spike_rect)
        elif self.type == "gap":
            gap_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            if player_rect.colliderect(gap_rect) and player.y + PLAYER_SIZE >= GROUND_HEIGHT - 5:
                return True
        elif self.type == "platform":
            platform_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            if (player_rect.top <= self.y + self.height and 
                player_rect.bottom > self.y + self.height and
                player_rect.right > self.x and player_rect.left < self.x + self.width and
                player.vel_y < 0):
                return True
            if (player_rect.bottom >= self.y and player_rect.bottom <= self.y + 15 and
                player_rect.right > self.x and player_rect.left < self.x + self.width and
                player.vel_y >= 0):
                return "platform"
        return False
    
    def is_offscreen(self):
        return self.x + self.width < 0

class ObstacleGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.obstacles = []
        self.distance = 0
        self.current_speed = SCROLL_SPEED
        
    def update(self):
        self.distance += self.current_speed
        
        speed_multiplier = 1 + min(self.distance / 20000, 0.5)
        self.current_speed = SCROLL_SPEED * speed_multiplier
        
        for obs in self.obstacles[:]:
            obs.x -= self.current_speed
            if obs.is_offscreen():
                self.obstacles.remove(obs)
        
        if len(self.obstacles) == 0 or self.obstacles[-1].x < WIDTH - self.get_current_spacing():
            self.spawn_obstacle()
    
    def get_current_spacing(self):
        difficulty = min(self.distance / 10000, 1.0)
        min_spacing = int(200 - 80 * difficulty)
        max_spacing = int(400 - 150 * difficulty)
        return random.randint(min_spacing, max_spacing)
    
    def spawn_obstacle(self):
        spacing = self.get_current_spacing()
        x = WIDTH if len(self.obstacles) == 0 else self.obstacles[-1].x + spacing
        
        difficulty = min(self.distance / 8000, 1.0)
        
        rand = random.random()
        if rand < 0.4:
            obs_type = "spike"
        elif rand < 0.7:
            obs_type = "gap"
        else:
            obs_type = "platform"
        
        if obs_type == "spike":
            self.obstacles.append(Obstacle(x, "spike"))
            if random.random() < 0.3 + difficulty * 0.4:
                self.obstacles.append(Obstacle(x + 50, "spike"))
                if random.random() < difficulty * 0.5:
                    self.obstacles.append(Obstacle(x + 100, "spike"))
        
        elif obs_type == "gap":
            gap_width = 100 + int(difficulty * 50)
            gap = Obstacle(x, "gap")
            gap.width = gap_width
            self.obstacles.append(gap)
            
            if random.random() < difficulty * 0.4:
                self.obstacles.append(Obstacle(x + gap_width + 20, "spike"))
        
        elif obs_type == "platform":
            self.obstacles.append(Obstacle(x, "platform"))
            
            if random.random() < 0.6:
                self.obstacles.append(Obstacle(x + 40, "spike"))
            
            if random.random() < difficulty * 0.5:
                self.obstacles.append(Obstacle(x + 150, "platform"))
    
    def check_collisions(self, player):
        for obs in self.obstacles:
            collision = obs.collides_with(player)
            if collision == "platform":
                player.y = obs.y - PLAYER_SIZE
                player.vel_y = 0
                player.on_ground = True
            elif collision:
                return True
        return False
    
    def draw(self, screen):
        for obs in self.obstacles:
            obs.draw(screen)
    
    def get_next_obstacles(self, n=5):
        """Get the next n obstacles ahead of the player"""
        visible_obstacles = [obs for obs in self.obstacles if obs.x > 100]
        return visible_obstacles[:n]


class ImpossibleGameEnv(gym.Env):
    """
    Gymnasium environment for the Impossible Game.
    
    Observation Space:
        - Player Y position (normalized)
        - Player Y velocity (normalized)
        - Player on ground (0 or 1)
        - Next 5 obstacles info (type, x, y, width) each normalized
    
    Action Space:
        - 0: Do nothing
        - 1: Jump
    
    Reward:
        - +1 for each frame survived
        - -100 for dying
        - +10 bonus for clearing an obstacle
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    
    def __init__(self, render_mode=None, max_steps=10000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Action space: 0 = no action, 1 = jump
        self.action_space = spaces.Discrete(2)
        
        # Observation space
        # [player_y, player_vel_y, on_ground, 
        #  obs1_type, obs1_x, obs1_y, obs1_width,
        #  obs2_type, obs2_x, obs2_y, obs2_width,
        #  ...] (5 obstacles)
        obs_size = 3 + 5 * 4  # 23 dimensions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize pygame
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Impossible Game - RL Training")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        self.player = None
        self.generator = None
        self.steps = 0
        self.score = 0
        self.obstacles_cleared = set()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.player = Player()
        self.generator = ObstacleGenerator(seed=seed)
        self.steps = 0
        self.score = 0
        self.obstacles_cleared = set()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Execute action
        if action == 1:
            self.player.jump()
        
        # Update game state
        self.player.update()
        self.generator.update()
        
        # Check for obstacles cleared (passed player x position)
        reward = 1.0  # Base reward for surviving
        for obs in self.generator.obstacles:
            obs_id = id(obs)
            if obs.x + obs.width < self.player.x and obs_id not in self.obstacles_cleared:
                self.obstacles_cleared.add(obs_id)
                reward += 10.0
        
        # Check collisions
        terminated = False
        if self.generator.check_collisions(self.player):
            self.player.dead = True
            terminated = True
            reward = -100.0  # Penalty for dying
        
        # Check if max steps reached
        truncated = self.steps >= self.max_steps
        
        self.score = int(self.generator.distance / 10)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """current observation state"""
        obs = np.zeros(23, dtype=np.float32)
        
        # Player state (normalized)
        obs[0] = (self.player.y - 0) / HEIGHT  # Player Y position
        obs[1] = self.player.vel_y / 20.0  # Player Y velocity (clamped)
        obs[2] = 1.0 if self.player.on_ground else 0.0
        
        # Next obstacles
        next_obstacles = self.generator.get_next_obstacles(5)
        
        obstacle_types = {"spike": 0.33, "gap": 0.66, "platform": 1.0}
        
        for i, obstacle in enumerate(next_obstacles):
            base_idx = 3 + i * 4
            obs[base_idx] = obstacle_types[obstacle.type]  # Type
            obs[base_idx + 1] = (obstacle.x - self.player.x) / WIDTH  # Relative X
            obs[base_idx + 2] = obstacle.y / HEIGHT  # Y position
            obs[base_idx + 3] = obstacle.width / WIDTH  # Width
        
        return obs
    
    def _get_info(self):
        return {
            'score': self.score,
            'distance': self.generator.distance,
            'speed': self.generator.current_speed,
            'steps': self.steps,
            'obstacles_cleared': len(self.obstacles_cleared)
        }
    
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
            self.clock.tick(FPS)
    
    def _render_frame(self):
        """Render the current game state"""
        if self.screen is None:
            return
        
        self.screen.fill(BLACK)
        
        # Draw background grid
        for i in range(0, WIDTH, 50):
            pygame.draw.line(self.screen, GRAY, (i, 0), (i, HEIGHT), 1)
        for i in range(0, HEIGHT, 50):
            pygame.draw.line(self.screen, GRAY, (0, i), (WIDTH, i), 1)
        
        # Draw ground
        pygame.draw.rect(self.screen, WHITE, (0, GROUND_HEIGHT, WIDTH, HEIGHT - GROUND_HEIGHT))
        
        # Draw obstacles
        self.generator.draw(self.screen)
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        speed_text = font.render(f"Speed: {self.generator.current_speed / SCROLL_SPEED:.1f}x", True, YELLOW)
        self.screen.blit(speed_text, (10, 50))
        
        steps_text = font.render(f"Steps: {self.steps}", True, WHITE)
        self.screen.blit(steps_text, (10, 90))
        
        pygame.display.flip()
    
    def close(self):
        if self.screen is not None:
            pygame.quit()


if __name__ == "__main__":
    # Example 1: Manual play test
    print("Testing environment with random actions...")
    env = ImpossibleGameEnv(render_mode="human")
    
    observation, info = env.reset()
    
    for _ in range(1000):
        # Random action (replace with your RL agent)
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished! Score: {info['score']}, Distance: {info['distance']:.0f}")
            observation, info = env.reset()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
    
    env.close()
    
    # Training with stable-baselines3
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    
    # Create and check environment
    env = ImpossibleGameEnv()
    check_env(env)
    
    # Train agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # Save model
    model.save("impossible_game_ppo")
    
    # Test trained agent
    env = ImpossibleGameEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    """