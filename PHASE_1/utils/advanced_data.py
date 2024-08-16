#eventually add unfiltered trajectories for training on failed trajectories too. so filter it in this script instead
from interceptor_generator import calculate_interception, plot_all_trajectories
from trajectory_generator_class import Projectile
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to calculate velocity and angle from position data
def calculate_velocity_and_angle(x_vals, y_vals, t_vals):
    velocities = []
    angles = []
    for i in range(1, len(t_vals)):
        dx = x_vals[i] - x_vals[i-1]
        dy = y_vals[i] - y_vals[i-1]
        dt = t_vals[i] - t_vals[i-1]
        velocity = np.sqrt(dx**2 + dy**2) / dt
        angle = np.arctan2(dy, dx)
        velocities.append(velocity)
        angles.append(np.degrees(angle))
    # Adding initial velocity and angle as the first values
    velocities.insert(0, velocities[0])
    angles.insert(0, angles[0])
    return np.array(velocities), np.array(angles)

# Function to detect collision
def check_collision(enemy_trajectory, interceptor_trajectory, tolerance=5):
    final_enemy_pos = enemy_trajectory[-1, :2]  # X, Y of the final point
    final_interceptor_pos = interceptor_trajectory[-1, :2]
    distance = np.linalg.norm(final_enemy_pos - final_interceptor_pos)
    return distance <= tolerance

# Function to plot the trajectory data
def plot_trajectory_data(trajectory_id, enemy_trajectory, interceptor_trajectory, enemy_velocities, enemy_angles, interceptor_velocities, interceptor_angles):
    plt.figure(figsize=(12, 8))

    # Plot positions
    plt.subplot(2, 2, 1)
    plt.plot(enemy_trajectory[:, 0], enemy_trajectory[:, 1], label='Enemy Projectile')
    plt.plot(interceptor_trajectory[:, 0], interceptor_trajectory[:, 1], label='Interceptor')
    plt.title(f'Trajectory {trajectory_id} - Position')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot velocities
    plt.subplot(2, 2, 2)
    plt.plot(enemy_trajectory[:, 2], enemy_velocities, label='Enemy Projectile Velocity')
    plt.plot(interceptor_trajectory[:, 2], interceptor_velocities, label='Interceptor Velocity')
    plt.title(f'Trajectory {trajectory_id} - Velocity')
    plt.xlabel('Timestep')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)

    # Plot angles
    plt.subplot(2, 2, 3)
    plt.plot(enemy_trajectory[:, 2], enemy_angles, label='Enemy Projectile Angle')
    plt.plot(interceptor_trajectory[:, 2], interceptor_angles, label='Interceptor Angle')
    plt.title(f'Trajectory {trajectory_id} - Angle')
    plt.xlabel('Timestep')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Set the CSV file path
csv_file = 'synthetic_trajectory_data.csv'

# Check if the file exists and read the last Trajectory_ID
if os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0:
    df_existing = pd.read_csv(csv_file)
    last_id = df_existing['Trajectory_ID'].max()
else:
    last_id = 0

for i in range(5000):
    data = []

    enemy_velocity = np.random.randint(100, 3501)
    enemy_launch_angle = np.random.randint(1, 90)
    print(f"Enemy velocity: {enemy_velocity} , Enemy launch angle: {enemy_launch_angle}")

    enemy = Projectile(enemy_velocity, enemy_launch_angle)
    enemy_trajectory = enemy.trajectory

    interceptor_trajectories, interceptor_angles = calculate_interception(enemy)

    if interceptor_trajectories:
        for trajectory in interceptor_trajectories:
            # Calculate velocities and angles for both enemy and interceptor
            enemy_velocities, enemy_angles = calculate_velocity_and_angle(enemy_trajectory[:, 0], enemy_trajectory[:, 1], enemy_trajectory[:, 2])
            interceptor_velocities, interceptor_angles = calculate_velocity_and_angle(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

            # Plotting the trajectories (uncomment to visualize)
            # plot_trajectory_data(i, enemy_trajectory, trajectory, enemy_velocities, enemy_angles, interceptor_velocities, interceptor_angles)

            # Check for collision and add a success indicator 1=success, 0=failed
            success = check_collision(enemy_trajectory, trajectory)

            # Save the data in the required format
            for timestep in range(len(trajectory)):
                row = [
                    last_id + 1,  # Trajectory ID
                    trajectory[timestep, 2],  # Timestep
                    enemy_trajectory[timestep, 0],  # Projectile Position X
                    enemy_trajectory[timestep, 1],  # Projectile Position Y
                    enemy_velocities[timestep],  # Projectile Velocity
                    enemy_angles[timestep],  # Projectile Angle
                    trajectory[timestep, 0],  # Interceptor Position X
                    trajectory[timestep, 1],  # Interceptor Position Y
                    interceptor_velocities[timestep],  # Interceptor Velocity
                    interceptor_angles[timestep],  # Interceptor Angle
                    int(success)  # Success Indicator (1 for success, 0 for failure)
                ]
                data.append(row)
            last_id += 1  # Increment ID after processing each trajectory

    else:
        print(f"Failed to intercept at iteration {i}.")

    df = pd.DataFrame(data, columns=[
        'Trajectory_ID', 'Timestep', 'Projectile_Position_X', 'Projectile_Position_Y', 
        'Projectile_Velocity', 'Projectile_Angle', 
        'Interceptor_Position_X', 'Interceptor_Position_Y', 
        'Interceptor_Velocity', 'Interceptor_Angle',
        'Success'  # Adding the success indicator
    ])

    # Check if the file exists
    if os.path.isfile(csv_file):
        # If the file exists, append without writing the header
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        # If the file does not exist, write the data with the header
        df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"{i} - Synthetic trajectory data saved to 'synthetic_trajectory_data.csv'.")
