from interceptor_generator import calculate_interception, plot_all_trajectories
from trajectory_generator_class import Projectile
import pandas as pd
import numpy as np


data = []

for i in range(5):
    enemy_velocity = np.random.randint(100, 3501)
    enemy_launch_angle = np.random.randint(1,90)
    print(f"Enemy velocity: {enemy_velocity} , Enemy launch angle: {enemy_launch_angle}")
    enemy = Projectile(enemy_velocity,enemy_launch_angle)
    filtered_trajectories, filtered_angles = calculate_interception(enemy)

    if filtered_trajectories:
        # plot_all_trajectories(enemy.trajectory, filtered_trajectories)
        for trajectory, angle in zip(filtered_trajectories, filtered_angles):
            data.append([enemy_velocity, enemy_launch_angle, 3000, angle,'Successful'])
    else:
        data.append([enemy_velocity, enemy_launch_angle, 3000,-1000,"Failed"])

df = pd.DataFrame(data, columns=['enemy_velocity', 'enemy_launch_angle','interception_velocity','interception_launch_angle','passed'])
df.to_csv('synthetic_data.csv',index=False)
print("Synthetic data saved to 'synthetic_data.csv'.")