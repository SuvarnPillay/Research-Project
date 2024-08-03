import sys
sys.path.insert(0, r'C:\Users\suvar\General\Work\Varsity\Honours\Research\Lab\Test1\utils')
from trajectory_generator_class import Projectile
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sin, cos
import numpy as np
from IPython.display import display as disp

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


def plot_both_trajectories(enemy_trajectory, interception_trajectories):

    
    plt.figure(figsize=(10, 6))  # Optional: set figure size

    x_vals = enemy_trajectory[:, 0]  # All rows, first column
    y_vals = enemy_trajectory[:, 1]  # All rows, second column
    

    
    plt.plot(x_vals, y_vals, label='Enemy Trajectory')

    for trajectory in interception_trajectories:
        # Assuming self.trajectory now contains your coordinates
        x_vals = trajectory[:, 0]  # All rows, first column
        y_vals = trajectory[:, 1]  # All rows, second column
        

       
        plt.plot(x_vals, y_vals, label='Interception Trajectory')

    plt.title('Projectile Trajectory')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.legend()  # Add the legend to the plot
    plt.grid(True)
    plt.show()


def calculate_interception(enemy: Projectile):
    MAX_VELOCITY = 2500
    INTERCEPTOR_START_X = 400000
    
    g_val = 9.81
    V0_P = enemy.params[0][0]
    Theta_P = enemy.params[0][1]
    V0_I = MAX_VELOCITY

    a = V0_P * np.cos(np.radians(Theta_P))
    b = V0_P * np.sin(np.radians(Theta_P))

    print("A:", a)
    print("B:", b)

    t, theta = sp.symbols('t theta')
    
    # Solving for launch angle first
    equation_theta = sp.Eq(b / MAX_VELOCITY, sp.sin(theta))
    Theta_I_Vals = sp.solve(equation_theta, theta)
    Theta_I_Vals = [theta for theta in Theta_I_Vals if 0 < theta < sp.pi]
    print("Theta_I_Vals:", Theta_I_Vals)

    # Solving for impact time
    impact_times = []
    for Theta_I in Theta_I_Vals:
        equation_time = sp.Eq(INTERCEPTOR_START_X + V0_I * sp.cos(Theta_I) * t, V0_P * sp.cos(np.radians(Theta_P)) * t)
        temp_times = sp.solve(equation_time, t)
        for time in temp_times:
            if time.is_real and time > 0:
                impact_times.append(time)

    print("Impact Times:", impact_times)
    if not impact_times:
        print("No valid impact times found.")
        return []

    impact_time = max(impact_times)

    print("Impact Time:", impact_time)

    trajectories = []

    for Theta_I in Theta_I_Vals:
        impact_time = float(impact_time)
        t_vals = np.linspace(0, impact_time, num=500)

        # Define symbolic variables
        v0, theta, g, t_sym = sp.symbols('v0 theta g t')

        # Define the equations
        x = v0 * sp.cos(theta) * t_sym + INTERCEPTOR_START_X
        y = v0 * sp.sin(theta) * t_sym - 0.5 * g * t_sym**2
        
        # Substitute values into the equations
        x_vals = [x.subs({v0: V0_I, theta: Theta_I, g: g_val, t_sym: ti}) for ti in t_vals]
        y_vals = [y.subs({v0: V0_I, theta: Theta_I, g: g_val, t_sym: ti}) for ti in t_vals]
        
        # Convert symbolic expressions to numerical values
        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        # Combine x and y into a single array of coordinates
        coordinates = np.column_stack((x_vals, y_vals, t_vals))
        trajectory = coordinates
        trajectories.append(trajectory)

    return trajectories




enemy = Projectile(2500,45)


interception_trajectories = calculate_interception(enemy)

if interception_trajectories:
    plot_both_trajectories(enemy.trajectory, interception_trajectories)



