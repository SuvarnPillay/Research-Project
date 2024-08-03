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

    # trajectories = [trajectory1, trajectory2]
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


def calculate_interception_attempt_1(enemy: Projectile):
    
    MAX_VELOCITY = 2500
    INTERCEPTOR_START_X = 400000
    
    g_val = 9.81
    V0_P = enemy.params[0][0]
    Theta_P = enemy.params[0][1]
    V0_I = MAX_VELOCITY

    a = V0_P * np.cos(np.radians(Theta_P))
    b = V0_P * np.sin(np.radians(Theta_P))

    print("A",a)
    print("B",b)

    #solving for impact time
    t, theta = sp.symbols('t theta')
    equation = sp.Eq(sp.asin(b/MAX_VELOCITY),sp.acos((a*t - INTERCEPTOR_START_X)/MAX_VELOCITY*t))
    impact_time = max(sp.solve(equation,t))
    print("impact time:",impact_time)

    #solving for launch angle
    equation = sp.Eq(impact_time, -INTERCEPTOR_START_X/(MAX_VELOCITY*sp.cos(theta) - a))
    disp(equation)

    #ADJUST THIS SO THAT IT ONLY TAKES VALUES BETWEEN 0 AND 180
    Theta_I = min(sp.solve(equation,theta))
    
    # denominator = V0_I*np.cos(np.arcsin(b/MAX_VELOCITY)) - a
    # print("Denominator",denominator)
    # t_until_impact = INTERCEPTOR_START_X / denominator
    # print("t_until_impact",t_until_impact)

    # Theta_I = np.arcsin(b/V0_I)

    print("Theta_I:", Theta_I)
    # print("Theta_I (degrees) =", np.degrees(Theta_I))

    # Theta_I_Radians = np.radians(Theta_I)

    # Calculate total flight time
    # t_flight = 2 * V0_I * np.sin(Theta_I) / g_val
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


    # If you want to combine x and y into a single array of coordinates
    coordinates = np.column_stack((x_vals, y_vals,t_vals))
    trajectory = coordinates
    print("Interceptors Coordinates:", trajectory)

    return trajectory


def calculate_interception(enemy: Projectile):
    
    MAX_VELOCITY = 2500
    INTERCEPTOR_START_X = 400000
    
    g_val = 9.81
    V0_P = enemy.params[0][0]
    Theta_P = enemy.params[0][1]
    V0_I = MAX_VELOCITY

    a = V0_P * np.cos(np.radians(Theta_P))
    b = V0_P * np.sin(np.radians(Theta_P))

    print("A",a)
    print("B",b)

    
    t, theta = sp.symbols('t theta')

    #solving for impact time
    equation = sp.Eq(sp.asin(b/MAX_VELOCITY),sp.acos((a*t - INTERCEPTOR_START_X)/MAX_VELOCITY*t))
    impact_time = max(sp.solve(equation,t))
    print("impact time:",impact_time)

    Y_Projectile = V0_P*np.sin(np.radians(Theta_P))*impact_time - (1/2) * g_val * impact_time**2
    print("Y_Projectile:", Y_Projectile)

    
    #solving for launch angle
    # equation = sp.Eq(impact_time, INTERCEPTOR_START_X/(MAX_VELOCITY*sp.cos(theta) - a))
    # disp(equation)

    equation = sp.Eq(Y_Projectile, V0_I*sp.sin(theta)*impact_time - (1/2)*g_val*impact_time**2)
    disp(equation)

    #ADJUST THIS SO THAT IT ONLY TAKES VALUES BETWEEN 0 AND 180
    Theta_I_Vals = sp.solve(equation,theta)
    
    print("Theta_I_Vals:", Theta_I_Vals)
    trajectories = []

    for Theta_I in Theta_I_Vals:
        # print("Theta_I (degrees) =", np.degrees(Theta_I))
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


        # If you want to combine x and y into a single array of coordinates
        coordinates = np.column_stack((x_vals, y_vals,t_vals))
        trajectory = coordinates
        trajectories.append(trajectory)
        # print("Interceptors Coordinates:", trajectory)

    return trajectories




enemy = Projectile(2500,45)

# plot_trajectory(enemy.trajectory)
# print(enemy.trajectory)

interception_trajectories = calculate_interception(enemy)
# plot_trajectory(interception_trajectory)
plot_both_trajectories(enemy.trajectory, interception_trajectories)


