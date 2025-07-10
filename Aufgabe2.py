from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
import matplotlib.animation as animation
import shapely.geometry as geo 

map = [[0,1000], [0,1000]]
target = (20, 30) #(20, 30)
init = (500,500)
obstacles = [[(300,0), (200,0), (200,500), (300,500)],[(650,0),(650, 300),(625,300),(625,0)],[(200,500),(200, 600),(500,600),(600,500)]]
#
#[[(100,0), (200,0), (200,140), (100,140)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(200,750),(200,850)]]
#[[(100,0), (200,0), (200,800), (100,800)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(400,750),(400,850)]]
#[[(100,0), (200,0), (200,140), (100,140)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(200,750),(200,850)]]

target_radius = 20


def create_transform(theta, dx, dy):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    transform = ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])
    return transform

def move2(polygon, dx, dy):
    temp = []
    for cords in polygon:
        temp.append((cords[0] + dx, cords[1] + dy))
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
        

def calcPath(robot, obstacles, target, map, steps, alg):
    PathHistory = [robot]
    PathHistoryMoves = [[(0, 0)]]
    PathHistoryDistance = [distanceToTarget(robot, target)]
    PathHistoryChecked = [False]
    moves = [(0, 1*steps), (0, -1*steps), (1*steps, 0), (-1*steps, 0)]      #(1*steps, 1*steps), (-1*steps, 1*steps), (1*steps, -1*steps), (-1*steps, -1*steps)
    i = 0
    j = 0
    counter = 0
    while(not collisionWithTarget(PathHistory[i], target) and counter < 10000000):
        if(not PathHistoryChecked[i]):
            for move in moves:
                temp = move2(PathHistory[i], move[0], move[1])
                if((not collision(temp, obstacles) and not(temp in PathHistory)) and isInMap(temp, map)):
                    lastMoves = PathHistoryMoves[i].copy()
                    lastMoves.append(move)
                    PathHistoryMoves.append(lastMoves)
                    PathHistory.append(temp)
                    PathHistoryDistance.append(distanceToTarget(temp, target))
                    PathHistoryChecked.append(False)
        PathHistoryChecked[i] = True
        i = alg(PathHistoryDistance, PathHistoryChecked, PathHistoryMoves, i)
        counter = counter + 1
    return PathHistoryMoves[i]

def distanceAlg(distance, checked, moves, i):
    distanceMin = 10000000
    distenceStepCount = 10000000
    stepsCount = 10000000
    tempDistance = i
    for pathIndex in range(0, len(checked)):
        if(not checked[pathIndex]):
            if((distance[pathIndex] < distanceMin)):
                distanceMin = distance[pathIndex]
                i = pathIndex 
            if(len(moves[pathIndex]) < stepsCount):
                stepsCount = len(moves[pathIndex])
                distenceStepCount = distance[pathIndex]
                j = pathIndex
            if(len(moves[pathIndex]) == stepsCount and (distance[pathIndex] < distenceStepCount)):
                distenceStepCount = distance[pathIndex]
                j = pathIndex
    if(distance[tempDistance] < distance[pathIndex]):
        i = j
    return i

def randomPick(distance, checked, moves, i):
    while(checked[i] == True):
        i = randrange(len(checked))
    return i

# Collision check funktioniert nur bei Rechtecken die nicht gedreht sind
def collision(robot, obstacles):
    robot_poly = geo.Polygon(robot)
    for obs in obstacles:
        obs_poly = geo.Polygon(obs)
        if robot_poly.intersects(obs_poly):
            return True
    return False

def collisionWithTarget(robot, target):
    robot_poly = geo.Polygon(robot)
    goal_circle = geo.Point(target).buffer(target_radius)
    return robot_poly.intersects(goal_circle)
    
def isInMap(robot, map):
    x, y = calcInit(robot)
    if((map[0][0]<= x <= map[0][1]) and (map[1][0] <= y <= map[1][1])):
        return True
    return False

def distanceToTarget(robot, target):
    xSum, ySum = calcInit(robot)
    return pow((int(xSum)-target[0]), 2) + pow((int(ySum)-target[1]), 2) 

def calcInit(robot):
    poly = geo.Polygon(robot)
    center = poly.centroid 
    return (center.x, center.y)

# Create a robot shaped
#robot = [(450, 450), (550, 450), (550, 550), (450, 550)]
robot = [(1000, 100), (1000, 200), (900, 200)] #[(900, 100), (1000, 100), (1000, 200), (900, 200)]
polygon_robot = create_polygon(robot)

init = calcInit(robot)

#Animation
t_stop = 1000
dt = 1
t = np.arange(0, t_stop, dt)
start = create_polygon(robot)

#create Path
moves = calcPath(robot, obstacles, target, map, 10, randomPick)

y = []
y.append(start)

for i in range(1, len(moves)):
    y.append(move(y[i - 1], 0, moves[i][0], moves[i][1]))


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
    return robot_patch, time_text


ani = animation.FuncAnimation(fig, animate, len(y), interval=50, blit=True)
plt.show()


# Create a robot shaped
#robot = [(450, 450), (550, 450), (550, 550), (450, 550)]
robot = [(1000, 100), (1000, 200), (900, 200)] #[(900, 100), (1000, 100), (1000, 200), (900, 200)]
polygon_robot = create_polygon(robot)

init = calcInit(robot)

#Animation
t_stop = 1000
dt = 1
t = np.arange(0, t_stop, dt)
start = create_polygon(robot)

#create Path
moves = calcPath(robot, obstacles, target, map, 10, distanceAlg)

y = []
y.append(start)

for i in range(1, len(moves)):
    y.append(move(y[i - 1], 0, moves[i][0], moves[i][1]))


fig, ax = visualize(map, obstacles, init, target, 0, "Abgabe 2.3")

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
start_x, start_y = y[0]
start_xy = np.column_stack((start_x, start_y))
robot_patch = Polygon(start_xy, closed=True, color='blue', alpha=0.3)
ax.add_patch(robot_patch)


def animate(i):
    time_text.set_text((i))
    xi, yi = y[i]
    robot_patch.set_xy(np.column_stack((xi, yi)))
    return robot_patch, time_text


ani = animation.FuncAnimation(fig, animate, len(y), interval=50, blit=True)
plt.show()
