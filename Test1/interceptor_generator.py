import sys
sys.path.insert(0, r'C:\Users\suvar\General\Work\Varsity\Honours\Research\Lab\Test1\utils')
from trajectory_generator_class import Projectile
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sin, cos

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




def calculate_interception(enemy: Projectile):
    
    MAX_VELOCITY = 2500

    a,b,V0_I, Theta_I, t = sp.symbols('a b Theta_I t')

    V0_P = enemy.params[0]
    Theta_P = enemy.params[1]
    V0_I = MAX_VELOCITY

    a = V0_P * sin(Theta_P)
    b = V0_P * cos(Theta_P)

    




enemy = Projectile(2500,45)

plot_trajectory(enemy.trajectory)
print(enemy.trajectory)

interception_trajectory = calculate_interception(enemy)



