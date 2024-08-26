import gym
from gym import spaces
import numpy as np

class InterceptionEnv(gym.Env):
    def __init__(self):
        super(InterceptionEnv, self).__init__()
        
        # Define action and observation space
        # Action space: [velocity adjustment, angle adjustment]
        self.action_space = spaces.Box(low=np.array([0, -10]), high=np.array([10, 10]), dtype=np.float32)
        
        # Observation space: [initial_velocity, initial_angle, projectile_x, projectile_y]
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.inf, -np.inf]), high=np.array([200, 90, np.inf, np.inf]), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        # Reset the environment state
        self.projectile_velocity = np.random.uniform(50, 150)  # Example values
        self.projectile_angle = np.random.uniform(20, 60)  # Example values
        self.interceptor_velocity = 0
        self.interceptor_angle = 0
        
        # Initial position of the projectile
        self.projectile_x = 0
        self.projectile_y = 0
        
        # Initial state
        return np.array([self.projectile_velocity, self.projectile_angle, self.projectile_x, self.projectile_y], dtype=np.float32)
    
    def step(self, action):
        velocity_adjustment, angle_adjustment = action
        
        # Update interceptor's velocity and angle
        self.interceptor_velocity += velocity_adjustment
        self.interceptor_angle += angle_adjustment
        
        # Update projectile position
        self.projectile_x += self.projectile_velocity * np.cos(np.radians(self.projectile_angle))
        self.projectile_y += self.projectile_velocity * np.sin(np.radians(self.projectile_angle)) - 0.5 * 9.81 * (self.projectile_x / self.projectile_velocity)**2
        
        # Calculate reward (example: reward for minimizing the distance to interception)
        distance_to_interceptor = np.sqrt((self.projectile_x - 0)**2 + (self.projectile_y - 0)**2)
        reward = -distance_to_interceptor
        
        # Done if projectile hits the ground
        done = self.projectile_y <= 0
        
        return np.array([self.projectile_velocity, self.projectile_angle, self.projectile_x, self.projectile_y], dtype=np.float32), reward, done, {}
    
    def render(self, mode='human'):
        pass



from stable_baselines3 import DQN

# Create the environment
env = InterceptionEnv()

# Initialize the DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("dqn_interceptor")


# Load the trained model
model = DQN.load("dqn_interceptor")

# Test the model
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
