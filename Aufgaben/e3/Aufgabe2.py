from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import math
import time
import pandas as pd
import numpy as np
import matplotlib.animation as animation



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
        self.grid = np.zeros((height, width), dtype=np.uint8)  # 0 = free, 1 = obstacle
        self.obstacles = []

    def set_obstacle(self, x, y):
        """
        Mark the cell (x, y) as occupied.
        """
        
        self.obstacles.append([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])
    #Add your code here

    def get_obstacles(self):
        return self.obstacles
    
    def is_free(self, x, y):
        """
        Check if the cell (x, y) is free.
        """
        for obstacle in self.obstacles:
            if(x == obstacle[0] and y == obstacle[1]):
                return False
        return True
    
    #Add your code here

    
    def visualize(self, start, target, title= "Environment"):
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
        
        start[0] = start[0] + 0.5
        start[1] = start[1] + 0.5
        target[0] = target[0] + 0.5
        target[1] = target[1] + 0.5

        ax.plot(start[0], start[1], 'go', label='start', markersize=10)
        ax.plot(target[0], target[1], 'ro', label='Goal', markersize=10)
        goal_circle = Circle(target, 1, color='red', alpha=0.3)

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.show()
        return fig, ax
    
    

grid = GridMap(10, 10, 1)
grid.set_obstacle(5, 6)
grid.visualize([5, 5], [8,8])