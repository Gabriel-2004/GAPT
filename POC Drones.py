import numpy as np
import random
import matplotlib.pyplot as plt

# Define the search space
SPACE_SIZE = 10
GRID_POINTS = np.arange(0, SPACE_SIZE + 1, 1)

# Initialize drone positions
drone1_pos = [0, 0, 5]
drone2_pos = [SPACE_SIZE, SPACE_SIZE, 10]  # Start drone2 at a higher altitude

# Search path (simple zigzag for demonstration)
def search_path(drone_pos, space_size, step=1, reverse=False):
    x, y, z = drone_pos
    if not reverse:
        if y % 2 == 0:
            if x < space_size:
                x += step
            else:
                y += step
        else:
            if x > 0:
                x -= step
            else:
                y += step
    else:
        if y % 2 == 0:
            if x > 0:
                x -= step
            else:
                y -= step
        else:
            if x < space_size:
                x += step
            else:
                y -= step
    return [x, y, z]

# Plotting the initial positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, SPACE_SIZE)
ax.set_ylim(0, SPACE_SIZE)
ax.set_zlim(0, 10)  # Adjust z-axis limit for altitude
ax.set_xticks(GRID_POINTS)
ax.set_yticks(GRID_POINTS)
ax.set_zticks(np.arange(0, 11, 1))  # Adjust z-axis ticks for altitude
ax.grid(True)

drone1_plot, = ax.plot([drone1_pos[0]], [drone1_pos[1]], [drone1_pos[2]], 'ro', label='Drone 1')
drone2_plot, = ax.plot([drone2_pos[0]], [drone2_pos[1]], [drone2_pos[2]], 'bo', label='Drone 2')

plt.legend()

# Simulation loop
reached_same_pos = False
while not reached_same_pos:
    drone1_pos = search_path(drone1_pos, SPACE_SIZE)
    temp_drone2_pos = search_path(drone2_pos, SPACE_SIZE, reverse=True)

    # Ensure drones don't collide or go out of bounds
    if drone1_pos[:2] == temp_drone2_pos[:2]:
        drone1_plot.set_data([drone1_pos[0]], [drone1_pos[1]])
        drone1_plot.set_3d_properties([drone1_pos[2]])
        print("Drone 1 finished at:", drone1_pos, "and drone 2 finished at:", drone2_pos)
        break
    drone2_pos = temp_drone2_pos    
    drone1_plot.set_data([drone1_pos[0]], [drone1_pos[1]])
    drone1_plot.set_3d_properties([drone1_pos[2]])
    drone2_plot.set_data([drone2_pos[0]], [drone2_pos[1]])
    drone2_plot.set_3d_properties([drone2_pos[2]])

    plt.draw()
    plt.pause(0.1)  # Pause to visualize movement

plt.show()
