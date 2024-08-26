import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the equations of motion including drag and wind
def projectile_motion(y, t, g, rho, Cd, A, wind_speed):
    x, vx, y, vy = y
    v = np.sqrt(vx**2 + vy**2)  # Velocity magnitude
    drag = 0.5 * rho * v**2 * Cd * A
    ax = -drag * vx / (rho * A)  # Acceleration in x due to drag
    ay = -g - (drag * vy / (rho * A))  # Acceleration in y due to drag and gravity
    ax += wind_speed  # Adding wind effect
    dydt = [vx, ax, vy, ay]
    return dydt

# Constants
x0 = 0
y0 = 0
vx0 = 100  # Initial velocity in x direction
vy0 = 50   # Initial velocity in y direction
g = 9.81   # Acceleration due to gravity
rho = 1.225  # Air density at sea level (kg/m^3)
Cd = 0.47  # Drag coefficient (typical for a sphere)
A = 0.01  # Cross-sectional area of projectile (m^2)
wind_speed = 5  # Wind speed (m/s), positive if in the direction of projectile

# Time points where solution is computed
t = np.linspace(0, 10, num=500)

# Solve ODE
y0 = [x0, vx0, y0, vy0]
solution = odeint(projectile_motion, y0, t, args=(g, rho, Cd, A, wind_speed))

# Extract results
x = solution[:, 0]
y = solution[:, 2]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Projectile Motion with Drag and Wind')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.show()
