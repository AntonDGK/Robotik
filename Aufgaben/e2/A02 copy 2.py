import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
import matplotlib.animation as animation

# Welt definieren
map_limits = [[0, 1000], [0, 1000]]
target = (20, 30)
init = (500, 500)
obstacles = [
    [(100, 0), (200, 0), (200, 140), (100, 140)],
    [(600, 1000), (600, 600), (700, 600), (700, 1000)],
    [(650, 0), (650, 300), (625, 300), (625, 0)],
    [(100, 850), (100, 750), (200, 750), (200, 850)]
]
target_radius = 20

# Bewegungsfunktion: schiebt das Polygon um dy nach oben
def pathAlg(t, polygon):
    return move(polygon, 0, 0, 1)

# Transformation erzeugen
def create_transform(theta, dx, dy):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])

# Transformation anwenden
def move(polygon, theta, dx, dy):
    transform = create_transform(theta, dx, dy)
    verticesX = []
    verticesY = []
    for i in range(len(polygon[0])):
        tempX = polygon[0][i] * transform[0][0] + polygon[1][i] * transform[0][1] + transform[0][2]
        verticesX.append(tempX)
        tempY = polygon[0][i] * transform[1][0] + polygon[1][i] * transform[1][1] + transform[1][2]
        verticesY.append(tempY)
    return (verticesX, verticesY)

# Startquadrat
robot = [(450, 450), (550, 450), (550, 550), (450, 550)]

# Hilfsfunktion
def create_polygon(vertices):
    arrayX = [v[0] for v in vertices] + [vertices[0][0]]
    arrayY = [v[1] for v in vertices] + [vertices[0][1]]
    return (arrayX, arrayY)

polygon_robot = create_polygon(robot)

# Visualisierung
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(map_limits[0])
ax.set_ylim(map_limits[1])
for obs in obstacles:
    ax.add_patch(Polygon(obs, closed=True, color='grey', alpha=0.7))
ax.plot(init[0], init[1], 'go', label='Start', markersize=10)
ax.plot(target[0], target[1], 'ro', label='Ziel', markersize=10)
goal_circle = Circle(target, target_radius, color='red', alpha=0.3)
ax.add_patch(goal_circle)
ax.set_title("Roboterbewegung")
ax.grid(True)
ax.legend()

# Zeit und Zustände
t_stop = 100
dt = 1
t = np.arange(0, t_stop, dt)
state = polygon_robot
y = [state]
for i in range(1, len(t)):
    y.append(pathAlg(t[i - 1], y[i - 1]))

# Roboter-Polygon initialisieren
start_x, start_y = y[0]
start_xy = np.column_stack((start_x, start_y))
robot_patch = Polygon(start_xy, closed=True, color='blue', alpha=0.7)
ax.add_patch(robot_patch)

# Zeittext
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# Animationsfunktion
def animate(i):
    time_text.set_text(f"t = {i*dt:.1f}s")
    xi, yi = y[i]
    robot_patch.set_xy(np.column_stack((xi, yi)))
    return robot_patch, time_text

# Animation starten
ani = animation.FuncAnimation(fig, animate, frames=len(y), interval=50, blit=False)
plt.show()