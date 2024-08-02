import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display as disp


class Projectile:
    def __init__(self,v0,theta):

        #attributes
        params_list = [
            (v0, theta)
        ]
        self.params = params_list
        self.trajectory = np.array([])

        # Define symbolic variables
        self.v0, self.theta, self.g, self.t = sp.symbols('v0 theta g t')

        # Define the equations
        self.x = self.v0 * sp.cos(self.theta) * self.t
        self.y = self.v0 * sp.sin(self.theta) * self.t - 0.5 * self.g * self.t**2

        # Display the equations
        disp(sp.Eq(sp.Symbol('x(t)'), self.x))
        disp(sp.Eq(sp.Symbol('y(t)'), self.y))

        self.create_trajectory()
   
        


    def create_trajectory(self):

        g_val = 9.81


        for v0_val,theta_val in self.params:
            # Convert angle to radians for calculation
            theta_rad_val = np.radians(theta_val)
            
            # Calculate total flight time
            t_flight = 2 * v0_val * np.sin(theta_rad_val) / g_val
            t_vals = np.linspace(0, t_flight, num=500)
            
            # Substitute values into the equations
            x_vals = [self.x.subs({self.v0: v0_val, self.theta: theta_rad_val, self.g: g_val, self.t: ti}) for ti in t_vals]
            y_vals = [self.y.subs({self.v0: v0_val, self.theta: theta_rad_val, self.g: g_val, self.t: ti}) for ti in t_vals]
            
            # Convert symbolic expressions to numerical values
            x_vals = np.array(x_vals, dtype=float)
            y_vals = np.array(y_vals, dtype=float)


            # If you want to combine x and y into a single array of coordinates
            coordinates = np.column_stack((x_vals, y_vals))
            self.trajectory = coordinates

            
            # # Create a legend string with all the information
            # legend_info = (
            #     f'Total Flight Time: {t_flight:.2f} s\n'
            #     f'Initial Velocity: {v0_val:.2f} m/s\n'
            #     f'Launch Angle: {theta_val:.2f} degrees'
            # )
            
            
# class Interceptor():

