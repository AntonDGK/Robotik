from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
import matplotlib.animation as animation
import shapely.geometry as geo                #Point, Polygon 
import random

class GridMap:
    #Your Code Goes in the sections Below
    def __init__(self, width, height, resolution=1):
        """
        Initialize a 2D occupancy grid map.
        :param width: Width of the map in number of cells.
        :param height: Height of the map in number of cells.
        :param resolution: Size of each cell (meters per cell).
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height * resolution, width * resolution), dtype=np.uint8)  # 0 = free, 1 = obstacle
        self.obstacles = []

    def set_grid(self, x, y):
        self.grid[x, y] = 1

    def set_obstacle(self, x, y):
        """
        Mark the cell (x, y) as occupied.
        """
        tempX = x * self.resolution
        tempY = y * self.resolution
        self.grid[int(tempX), int(tempY)] = 1
        
        self.obstacles.append([(x, y), ((x+1/self.resolution), y), ((x+1/self.resolution), (y+1/self.resolution)), (x, ((y+1/self.resolution)))])

    def set_polygonWithoutGrid(self, polygon):
        self.obstacles.append(polygon)

    def get_obstacles(self):
        return self.obstacles

    def is_freeGrid(self,x,y):
        return self.grid[x,y]

    def is_free(self, x, y):
        """
        Check if the cell (x, y) is free.
        """
        if(x < 0 or x >= self.width * self.resolution or y < 0 or y >= self.height * self.resolution):
            return False
        if(self.grid[x, y] == 1):
            return False
        return True

    def visualizeWithoutShow(self, start, target, title= "Environment"):
        """
        Show the grid map using matplotlib, with grid lines.
        """
        #Add your code here
        fig, ax = plt.subplots(figsize=(8,8))

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])

        for obstacle in self.obstacles:
            polygon = Polygon(obstacle, closed=True, color='grey', alpha=0.7)
            ax.add_patch(polygon)
        
        start[0] = start[0] + 0.5/self.resolution
        start[1] = start[1] + 0.5/self.resolution
        target[0] = target[0] + 0.5/self.resolution
        target[1] = target[1] + 0.5/self.resolution

        ax.plot(start[0], start[1], 'go', label='start', markersize=min(10, 25/self.resolution))
        ax.plot(target[0], target[1], 'ro', label='Goal', markersize=min(10, 25/self.resolution))
        goal_circle = Circle(target, 1, color='red', alpha=0.3)

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        return fig, ax
    
    def visualize(self, start, target):
        self.visualizeWithoutShow(start, target)
        plt.show()

    def connectPoints(self, points):
        output1 = []
        output2 = []
        for point in points:
            output1.append(point[0])
            output2.append(point[1])
        return [output1, output2]
    
    def create_polygon_sizeResolution(self, cords):
        temp = [[cords[0]/self.resolution, cords[1]/self.resolution], [cords[0]/self.resolution + 1/self.resolution, cords[1]/self.resolution], 
                [cords[0]/self.resolution + 1/self.resolution, cords[1]/self.resolution + 1/self.resolution], [cords[0]/self.resolution, cords[1]/self.resolution + 1/self.resolution], 
                [cords[0]/self.resolution, cords[1]/self.resolution]]
        return self.connectPoints(temp)

    def moveToLine(self, start, moves):
        start = [start[0] + 1/2*self.resolution, start[1] + 1/2*self.resolution]
        temp =[start]
        for i in range(1, len(moves)):
            temp.append([temp[i-1][0] + moves[i][0]/self.resolution, temp[i-1][1] + moves[i][1]/self.resolution])
        return temp

    def dividRes(self, array):
        out = []
        for i in array:
            out.append(i/self.resolution)
        return out

    def animate(self, start, goal, calcPath):
        #Animation
        t_stop = 1000
        dt = 1
        t = np.arange(0, t_stop, dt)
        startRobot = (start[0], start[1]), ((start[0]+1/self.resolution), start[1]), ((start[0]+1/self.resolution), (start[1]+1/self.resolution)), (start[0], ((start[1]+1/self.resolution)))

        #create Path
        moves = calcPath(self, start, goal)
        
        start_cords = [[int(start[0]*self.resolution), int(start[1]*self.resolution)]]
        robot = [[int(start[0]*self.resolution), int(start[1]*self.resolution)]]
        y = []
        y.append(self.create_polygon_sizeResolution(robot[0]))
        for i in range(1, len(moves)):
            robot.append([robot[i-1][0] + moves[i][0], robot[i-1][1] + moves[i][1]])
            y.append(self.create_polygon_sizeResolution(robot[i]))
            
        fig, ax = self.visualizeWithoutShow(start, goal)
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        start_x, start_y = y[0]
        start_xy = np.column_stack((start_x, start_y))
        robot_patch = Polygon(start_xy, closed=True, color='blue', alpha=0.3)
        ax.add_patch(robot_patch)
        
        for i in y:
            xi, yi = i
            polygon = Polygon(np.column_stack((xi, yi)), closed=True, color='green', alpha=0.2)
            ax.add_patch(polygon)  
        
        dynamisch = []
        for i in range(0, len(robot)):
            dynamisch.append((self.dynamic_obstacle(ax, [robot[i][0], robot[i][1]])))
            if(dynamisch[i] != None):
                ax.add_patch(dynamisch[i])
        k = 0
        n = 0
        p = 0
        while(k < len(robot) - 1):
            p = 0
            while(self.is_freeGrid(robot[k+p][0], robot[k+p][1])):
                p = p + 1
            if(0 < p):
                movesBetween = calcPath(self, self.dividRes(robot[k-1]), self.dividRes(robot[k+p]))
                if(movesBetween != None):
                    moves = moves[0:(n)] + movesBetween + moves[(n+p+1):]
                    n = n + len(movesBetween)-1
                    k = k+p
            k = k+1
            n = n+1
            



        robot = [[int(start[0]*self.resolution), int(start[1]*self.resolution)]]
        y = []
        y.append(self.create_polygon_sizeResolution(robot[0]))
        for i in range(1, len(moves)):
            robot.append([robot[i-1][0] + moves[i][0], robot[i-1][1] + moves[i][1]])
            y.append(self.create_polygon_sizeResolution(robot[i]))


        def animate(i):
            time_text.set_text((i))
            xi, yi = y[i]
            robot_patch.set_xy(np.column_stack((xi, yi)))

            b = False
            for j in range(0, len(dynamisch)):
                if(dynamisch[j] != None):
                    if(distance_PolygonSameSize(robot_patch, dynamisch[j]) > 6/self.resolution):
                        dynamisch[j].set_color("none")
                    else:
                        dynamisch[j].set_color("red")
                    
            dynamisch_animated = dynamisch.copy()
            while(None in dynamisch_animated):
                dynamisch_animated.remove(None)


            dynamisch_animated.append(robot_patch)
            dynamisch_animated.append(time_text)
            return dynamisch_animated
                
        ani = animation.FuncAnimation(fig, animate, len(y), interval=50, blit=True)
        plt.show()


    def dynamic_obstacle(self, ax, robot_pos, chance=0.3, radius=2):    #true_map in self
        if(random.random() < chance):
            x = random.randrange(radius*2) - radius
            y = random.randrange(radius*2) - radius
            if(self.grid[robot_pos[0] + x, robot_pos[1] + y] == 0):
                self.grid[robot_pos[0] + x, robot_pos[1] + y] = 1
                x = x + robot_pos[0]
                y = y + robot_pos[1]
                self.set_grid(x, y)
                x = x / self.resolution
                y = y / self.resolution
                temp = ([(x, y), ((x+1/self.resolution), y), ((x+1/self.resolution), (y+1/self.resolution)), (x, ((y+1/self.resolution)))])
                return Polygon(temp, closed=True, color='red', alpha=0.7)
            return None
        return None
    

def a_star(gridmap, robot, target, steps = 1):   
    PathHistory = [[int(robot[0]*gridmap.resolution), int(robot[1]*gridmap.resolution)]]
    targetResult = [target[0]*gridmap.resolution, target[1]*gridmap.resolution]
    PathHistoryMoves = [[(0, 0)]]
    moves = [(0, 1*steps), (0, -1*steps), (1*steps, 0), (-1*steps, 0)]      #(1*steps, 1*steps), (-1*steps, 1*steps), (1*steps, -1*steps), (-1*steps, -1*steps)
    i = 0
    temp = []
    while(not(PathHistory[i] == targetResult) and (i < 1000*gridmap.resolution*gridmap.resolution)):
        for move in moves:
            temp = [PathHistory[i][0] + move[0], PathHistory[i][1] + move[1]]
            if(gridmap.is_free(temp[0], temp[1]) and not(temp in PathHistory)):
                lastMoves = PathHistoryMoves[i].copy()
                lastMoves.append(move)
                PathHistoryMoves.append(lastMoves)
                PathHistory.append(temp)
        i = i + 1
        if(i >= len(PathHistory)):
            return None
    return PathHistoryMoves[i]

def distance_cordsSameSize(poly1, poly2):
    X1 = 100
    X2 = 100
    Y1 = 100
    Y2 = 100
    for i in poly1:
        X1 = min(i[0], X1)
        Y1 = min(i[1], Y1)
    for j in poly2:
        X2 = min(j[0], X2)
        Y2 = min(j[1], Y2)
    return abs(X1-X2) + abs(Y1-Y2)

def distance_PolygonSameSize(poly1, poly2):
    return distance_cordsSameSize(poly1.xy, poly2.xy)


def add_polygon_obstacle_to_grid(gridmap, polygon, origin=(0, 0)):
    """
    Rasterize a polygonal obstacle into the gridmap.
    :param gridmap: Instance of GridMap.
    :param polygon: A shapely.geometry.Polygon object.
    :param origin: Origin of world coordinates (for future use).
    """
    #Add your code here
    #gridmap.set_polygonWithoutGrid(polygon.exterior.coords)
    for y in range(0, gridmap.height*gridmap.resolution):
        tempY = y/gridmap.resolution
        for x in range(0, gridmap.width*gridmap.resolution):
            tempX = x/gridmap.resolution
            p = geo.Point((tempX + 1/(2*gridmap.resolution), tempY + 1/(2*gridmap.resolution)))
            if (polygon.contains(p) or polygon.touches(p)):
                #gridmap.set_grid(x,y)
                gridmap.set_obstacle(x/gridmap.resolution ,y/gridmap.resolution)


grid = GridMap(10, 10, 5)
grid.set_obstacle(1, 1)

print("Is (4, 2) free?", grid.is_free(4, 2))  # False
print("Is (1, 1) free?", grid.is_free(1, 1))  # True

start = [5, 5]
goal = [1.4, 1.4]

square = geo.Polygon([(1.5, 0.0), (2.5, 0.0), (2.5, 0.9), (1.5, 0.9)])
triangle = geo.Polygon([(2.0, 1.0), (3.0, 2.5), (4.0, 1.0)])
l_shape = geo.Polygon([(0.0, 3.0), (6.0, 3.0), (6.0, 1.0), (7.0, 1.0), (7.0, 4.0), (0.0, 4.0)])

add_polygon_obstacle_to_grid(grid, square)
add_polygon_obstacle_to_grid(grid, triangle)
add_polygon_obstacle_to_grid(grid, l_shape)

#a_star(grid, start, goal)
#grid.visualize(start, goal)

grid.animate(start, goal, a_star)