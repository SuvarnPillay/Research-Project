import numpy as np
import matplotlib.pyplot as plt
from trajectory_generator_class import Projectile

def calculate_interception_using_controls(enemy: Projectile, controls, dt=0.1, max_time=50):
    INTERCEPTOR_START_X = 400000
    INTERCEPTOR_START_Y = 0
    g_val = 9.81
    
    interceptor_position = np.array([INTERCEPTOR_START_X, INTERCEPTOR_START_Y])
    interceptor_velocity = np.array([0.0, 0.0])

    trajectories = []
    
    time_steps = np.arange(0, max_time, dt)
    for ax, ay in controls:
        trajectory = []
        for t in time_steps:
            # Update velocity
            interceptor_velocity[0] += ax * dt
            interceptor_velocity[1] += ay * dt - g_val * dt
            
            # Update position
            interceptor_position += interceptor_velocity * dt
            
            # Save the position to the trajectory
            trajectory.append(interceptor_position.copy())
            
            # Check if the interceptor is within a certain tolerance of the enemy
            enemy_x = enemy.params[0][0] * np.cos(np.radians(enemy.params[0][1])) * t
            enemy_y = enemy.params[0][0] * np.sin(np.radians(enemy.params[0][1])) * t - 0.5 * g_val * t ** 2
            
            if np.linalg.norm(interceptor_position - np.array([enemy_x, enemy_y])) < 10:  # Adjust the tolerance as needed
                trajectories.append(np.array(trajectory))
                break
    return trajectories

def plot_trajectories(trajectories, enemy_trajectory):
    plt.figure(figsize=(10, 6))
    
    for trajectory in trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Interception Trajectory')
    
    plt.plot(enemy_trajectory[:, 0], enemy_trajectory[:, 1], label='Enemy Trajectory', color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
enemy = Projectile(100, 45)

# Example controls: (ax, ay) values in m/s^2
controls = [(10, 15), (12, 12), (15, 10)]  # You can expand this list or use different control strategies

# Calculate the interception trajectory
trajectories = calculate_interception_using_controls(enemy, controls)

# Assuming you have the enemy trajectory from somewhere
enemy_trajectory = enemy.trajectory

# Plot the results
plot_trajectories(trajectories, enemy_trajectory)
