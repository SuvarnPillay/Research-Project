import sys
sys.path.insert(0, r'C:\Users\suvar\General\Work\Varsity\Honours\Research\Lab\Test1\utils')
from trajectory_generator_class import Projectile
import matplotlib.pyplot as plt

def plot_trajectory(trajectory):

    # Assuming self.trajectory now contains your coordinates
    x_vals = trajectory[:, 0]  # All rows, first column
    y_vals = trajectory[:, 1]  # All rows, second column
    

    plt.figure(figsize=(10, 6))  # Optional: set figure size
    plt.plot(x_vals, y_vals, label='Trajectory')

    plt.title('Projectile Trajectory')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.legend()  # Add the legend to the plot
    plt.grid(True)
    plt.show()


enemy = Projectile(2500,45)

plot_trajectory(enemy.trajectory)



def calculate_interception():
