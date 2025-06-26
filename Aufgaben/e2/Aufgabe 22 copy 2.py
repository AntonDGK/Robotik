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



def pathAlg(t, polygon):
    if(t < t_stop/4):
        return move(polygon, 0, 0, 5)
    else:
        return move(polygon, 0, 5, 0)

def create_transform(theta, dx, dy):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    transform = ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])
    return transform

def move2(polygon, dx, dy):
    temp = [(polygon[0][0] + dx, polygon[0][1] + dy), (polygon[1][0] + dx, polygon[1][1] + dy), (polygon[2][0] + dx, polygon[2][1] + dy) , (polygon[3][0] + dx, polygon[3][1] + dy)]
    #for i in range(0, len(polygon)):
    #    print((polygon[i][0] + dx, polygon[i][1] + dy))
    #    temp.append(polygon[i][0] + dx, polygon[i][1] + dy)
    return temp

def move(polygon, theta, dx, dy):
    transform = create_transform(theta, dx, dy)
    verticesX = []
    verticesY = []
    for i in range(len(polygon[0])):
        tempX = polygon[0][i] * transform[0][0] + polygon[1][i] * transform[0][1] + transform[0][2]
        tempX = int(tempX)
        verticesX.append(tempX)
        tempY = polygon[0][i] * transform[1][0] + polygon[1][i] * transform[1][1] + transform[1][2]
        tempY = int(tempY)
        verticesY.append(tempY)
    return (verticesX, verticesY)


def visualize(map_limits, obstacles, start, target, movable, title= "Environment"):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_xlim(map_limits[0])
    ax.set_ylim(map_limits[1])

    for obs in obstacles:
         polygon = Polygon(obs, closed=True, color='grey', alpha=0.7)
         ax.add_patch(polygon)
    
    ax.plot(start[0], start[1], 'go', label='start', markersize=10)
    ax.plot(target[0], target[1], 'ro', label='Goal', markersize=10)
    goal_circle = Circle(target, target_radius, color='red', alpha=0.3)

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig, ax

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
        
        plt.fill(polygons[i][0], polygons[i][1], label=label, edgecolor=color, facecolor=color, alpha=0.3)
        plt.legend()
    return ax
        

# Collision check funktioniert nur bei Rechtecken die nicht gedreht sind
def collision(robot, obstacles):
    xSum = 0
    ySum = 0
    maxX = 0
    maxY = 0 
    for r in robot:
        xSum = xSum + r[0]
        ySum = ySum + r[1]
        maxX = max(maxX, r[0])
        maxY = max(maxY, r[1])
    xSum = xSum/4
    ySum = ySum/4
    maxX = maxX - xSum
    maxY = maxY - ySum
    for obstacle in obstacles:
        if((obstacle[0][0] - maxX <= xSum <= obstacle[2][0] + maxX) and (obstacle[0][1] - maxY <= ySum <= obstacle[2][1] + maxY)):
            return True
    return False
    
def isInMap(robot, map):
    xSum = 0
    ySum = 0
    maxX = 0
    maxY = 0 
    for r in robot:
        xSum = xSum + r[0]
        ySum = ySum + r[1]
        maxX = max(maxX, r[0])
        maxY = max(maxY, r[1])
    xSum = xSum/4
    ySum = ySum/4
    maxX = maxX - xSum
    maxY = maxY - ySum
    if((map[0][0] - maxX <= xSum <= map[0][1] + maxX) and (map[1][0] - maxY <= ySum <= map[1][1] + maxY)):
        return True
    return False

def collisionWithTarget(robot, target):
    target
    if((robot[0][0] <= target[0] <= robot[2][0]) and (robot[0][1] <= target[1] <= robot[2][1])):
        return True
    return False

def calcPath(robot, obstacles, target, map, steps):
    PathHistory = [robot]
    PathHistoryMoves = [[(0, 0)]]
    moves = [(0, 1*steps), (0, -1*steps), (1*steps, 0), (-1*steps, 0)]      #(1*steps, 1*steps), (-1*steps, 1*steps), (1*steps, -1*steps), (-1*steps, -1*steps)
    i = 0
    while(not collisionWithTarget(PathHistory[i], target) and i < 10000000):
        for move in moves:
            temp = move2(PathHistory[i], move[0], move[1])
            if((not collision(temp, obstacles) and not(temp in PathHistory)) and isInMap(temp, map)):
                lastMoves = PathHistoryMoves[i].copy()
                lastMoves.append(move)
                PathHistoryMoves.append(lastMoves)
                PathHistory.append(temp)
        print(i)
        i = i + 1
    print(PathHistory[i])
    print(lastMoves)
    return PathHistoryMoves[i]


# Create a robot shaped
robot = [(450, 450), (550, 450), (550, 550), (450, 550)]
polygon_robot = create_polygon(robot)



#Animation
t_stop = 1000
dt = 1
t = np.arange(0, t_stop, dt)
start = create_polygon(robot)

#create Path
moves = calcPath(robot, obstacles, target, map, 10)



y = []
y.append(start)

for i in range(1, len(moves)):
    y.append(move(y[i - 1], 0, moves[i][0], moves[i][1]))


#for i in range(1, len(t)):
#    y.append(pathAlg(t[i - 1], y[i - 1]))
#    if(y[i - 1] == pathAlg(i, y[i])):
#        break

fig, ax = visualize(map, obstacles, init, target, 0, "Abgabe 2.2")

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
start_x, start_y = y[0]
start_xy = np.column_stack((start_x, start_y))
robot_patch = Polygon(start_xy, closed=True, color='blue', alpha=0.3)
ax.add_patch(robot_patch)


def animate(i):
    time_text.set_text((i))
    xi, yi = y[i]
    robot_patch.set_xy(np.column_stack((xi, yi)))
    #print(y[i])
    return robot_patch, time_text


ani = animation.FuncAnimation(fig, animate, len(y), interval=50, blit=True)
plt.show()
