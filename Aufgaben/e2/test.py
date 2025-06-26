import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
import math

# Map definition
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

# Robot definition
robot = [(0, 0), (50, 0), (50, 50), (0, 50)]  # centered at origin

# --- Minkowski Sum using Star Algorithm ---
def minkowski_sum(obs, robot):
    # Invert robot to compute O ⊖ A
    robot = [(-x, -y) for (x, y) in robot]

    def sort_by_angle(polygon):
        center = np.mean(polygon, axis=0)
        return sorted(polygon, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))

    A = sort_by_angle(obs)
    B = sort_by_angle(robot)

    result = []
    i = j = 0
    len_a = len(A)
    len_b = len(B)

    while i < len_a or j < len_b:
        a = A[i % len_a]
        b = B[j % len_b]
        result.append((a[0] + b[0], a[1] + b[1]))

        next_a = A[(i + 1) % len_a]
        next_b = B[(j + 1) % len_b]
        vec_a = (next_a[0] - a[0], next_a[1] - a[1])
        vec_b = (next_b[0] - b[0], next_b[1] - b[1])
        angle_a = math.atan2(vec_a[1], vec_a[0])
        angle_b = math.atan2(vec_b[1], vec_b[0])

        diff = (angle_a - angle_b + 2 * math.pi) % (2 * math.pi)
        if abs(diff) < 1e-6:
            i += 1
            j += 1
        elif diff < math.pi:
            i += 1
        else:
            j += 1

        if len(result) > len_a + len_b:  # safety to avoid infinite loops
            break

    return result

# --- Visualization ---
def visualize_environment():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(map_limits[0])
    ax.set_ylim(map_limits[1])
    ax.set_title("C_obs via Minkowski Difference")
    ax.set_aspect('equal')

    # Draw workspace obstacles
    for obs in obstacles:
        patch = Polygon(obs, closed=True, color='grey', alpha=0.4, label='Workspace Obstacle')
        ax.add_patch(patch)

    # Draw C_obs regions
    for obs in obstacles:
        c_obs = minkowski_sum(obs, robot)
        c_obs.append(c_obs[0])  # close polygon
        x, y = zip(*c_obs)
        ax.fill(x, y, color='red', alpha=0.3, label='C_obs')

    # Draw target
    ax.plot(target[0], target[1], 'ro', label='Goal', markersize=10)
    goal_circle = Circle(target, target_radius, color='red', alpha=0.3)
    ax.add_patch(goal_circle)

    # Draw initial robot position (centered at init)
    translated_robot = [(x + init[0], y + init[1]) for (x, y) in robot]
    x, y = zip(*translated_robot + [translated_robot[0]])
    ax.fill(x, y, color='blue', alpha=0.4, label='Robot Init')

    ax.legend()
    ax.grid(True)
    plt.show()

# --- Run ---
visualize_environment()