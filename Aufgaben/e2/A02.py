from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import math
import time
import pandas as pd
import numpy as np
import matplotlib.animation as animation

map = [[0,1000], [0,1000]]
target = (20, 30)
init = (500,500)
obstacles = [[(100,0), (200,0), (200,140), (100,140)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(200,750),(200,850)]]
target_radius = 20

#Animation
t = 100

def pathAlg(t, polygon):
    return

def create_transform(theta, dx, dy):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    transform = ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])
    return transform

def move(t, polygon, theta, dx, dy):
    transform = create_transform(theta, dx, dy)
    verticesX = []
    verticesY = []
    for i in range(len(polygon[0])):
        tempX = polygon[0][i] * transform[0][0] + polygon[1][i] * transform[0][1] + transform[0][2]
        verticesX.append(tempX)
        tempY = polygon[0][i] * transform[1][0] + polygon[1][i] * transform[1][1] + transform[1][2]
        verticesY.append(tempY)
    return (verticesX, verticesY)


def visualize(map_limits, obstacles, start, target, movable, terget_radius, title= "Environment"):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_xlim(map_limits[0])
    ax.set_ylim(map_limits[1])

    for obs in obstacles:
         polygon = Polygon(obs, closed=True, color='grey', alpha=0.7)
         ax.add_patch(polygon)
    
    plot_polygons(movable, colors=['blue'], labels=['Bot'])
    ax.plot(start[0], start[1], 'go', label='start', markersize=10)
    ax.plot(target[0], target[1], 'ro', label='Goal', markersize=10)
    goal_circle = Circle(target, target_radius, color='red', alpha=0.3)

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    #plt.show()
    return fig

def create_polygon(vertices):

    arrayX = []
    arrayY = []
    for i in vertices:
        arrayX.append(i[0])
        arrayY.append(i[1])
    arrayX.append(vertices[0][0])
    arrayY.append(vertices[0][1])

         # Create a figure containing a single Axes.            
    return (arrayX, arrayY)

def plot_polygons(polygons, colors=None, labels=None, title="Robot Bodies"):
    #plt.figure(figsize=(6, 6))
    
    for i, poly in enumerate(polygons):
        closed_poly = np.vstack([poly, poly[0]])
        color = colors[i] if colors and i < len(colors) else 'blue'
        label = labels[i] if labels and i < len(labels) else f"Polygon {i+1}"

        # Add code here
        
        pol = plt.fill(polygons[i][0], polygons[i][1], label=label, edgecolor=color, facecolor=color, alpha=0.3)
        plt.legend()
        
    
# Create a triangle shaped



robot = [(450, 450), (550, 450), (550, 550), (450, 550)]
polygon_robot = create_polygon(robot)

fig = visualize(map, obstacles, init, target, [polygon_robot], 0, "test")
#ani = animation.FuncAnimation(fig, 0, 1000, interval=t*1000, blit=True)
plt.show()
#visualize(map, obstacles, init, target, target_radius)