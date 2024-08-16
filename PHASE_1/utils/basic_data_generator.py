from interceptor_generator import calculate_interception, plot_all_trajectories
from trajectory_generator_class import Projectile
import pandas as pd
import numpy as np
import os




for i in range(2):

    data = []
    
    enemy_velocity = np.random.randint(100, 3501)
    enemy_launch_angle = np.random.randint(1,90)
    print(f"Enemy velocity: {enemy_velocity} , Enemy launch angle: {enemy_launch_angle}")

    enemy = Projectile(enemy_velocity,enemy_launch_angle)
    enemy_trajectory = enemy.trajectory
    print(f"Enemy trajectory: {enemy_trajectory}")

    #eventually add unfiltered trajectories for training on failed trajectories too. so filter it in this script instead
    interceptor_trajectories, interceptor_angles = calculate_interception(enemy)

    if interceptor_trajectories:
        # plot_all_trajectories(enemy.trajectory, interceptor_trajectories)
        for trajectory, angle in zip(interceptor_trajectories, interceptor_angles):
            data.append([enemy_velocity, enemy_launch_angle, 3000, angle,'Successful'])
    else:
        data.append([enemy_velocity, enemy_launch_angle, 3000,-1000,"Failed"])

    df = pd.DataFrame(data, columns=['enemy_velocity', 'enemy_launch_angle','interception_velocity','interception_launch_angle','passed'])
    # Set the CSV file path
    csv_file = 'synthetic_data.csv'

    # Check if the file exists
    if os.path.isfile(csv_file):
        # If the file exists, append without writing the header
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        # If the file does not exist, write the data with the header
        df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"{i} - Synthetic data saved to 'synthetic_data.csv'.")
