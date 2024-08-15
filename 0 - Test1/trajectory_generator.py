import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display as disp


def plot_trajectory(params):

    g_val = 9.81
    # Plot the trajectories
    plt.figure(figsize=(10, 5))

    for v0_val,theta_val in params:
        # Convert angle to radians for calculation
        theta_rad_val = np.radians(theta_val)
        
        # Calculate total flight time
        t_flight = 2 * v0_val * np.sin(theta_rad_val) / g_val
        t_vals = np.linspace(0, t_flight, num=500)
        
        # Substitute values into the equations
        x_vals = [x.subs({v0: v0_val, theta: theta_rad_val, g: g_val, t: ti}) for ti in t_vals]
        y_vals = [y.subs({v0: v0_val, theta: theta_rad_val, g: g_val, t: ti}) for ti in t_vals]
        
        # Convert symbolic expressions to numerical values
        x_vals = [float(xi) for xi in x_vals]
        y_vals = [float(yi) for yi in y_vals]

        # # Create a legend string with all the information
        # legend_info = (
        #     f'Total Flight Time: {t_flight:.2f} s\n'
        #     f'Initial Velocity: {v0_val:.2f} m/s\n'
        #     f'Launch Angle: {theta_val:.2f} degrees'
        # )
        
        
        # plt.plot(x_vals, y_vals, label=legend_info )
        plt.plot(x_vals, y_vals)

    plt.title('Projectile Trajectory')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.legend()  # Add the legend to the plot
    plt.grid(True)
    plt.show()




# Define symbolic variables
v0, theta, g, t = sp.symbols('v0 theta g t')

# Define the equations
x = v0 * sp.cos(theta) * t
y = v0 * sp.sin(theta) * t - 0.5 * g * t**2

# Display the equations
disp(sp.Eq(sp.Symbol('x(t)'), x))
disp(sp.Eq(sp.Symbol('y(t)'), y))

# Example usage with different parameters
params_list = [
    (2500, 45),
    (2400, 30),
    (3000, 60)
]
plot_trajectory(params_list)
