from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import math
import time
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import shapely.geometry as geo                #Point, Polygon 


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
        
    def set_grid(self, x, y):
        self.grid[x, y] = 1

    def set_obstacle(self, x, y):
        """
        Mark the cell (x, y) as occupied.
        """
        self.grid[x, y] = 1
        self.obstacles.append([(x, y), (x+self.resolution, y), (x+self.resolution, y+self.resolution), (x, y+self.resolution)])

    def set_polygonWithoutGrid(self, polygon):
        self.obstacles.append(polygon)

    def get_obstacles(self):
        return self.obstacles
    
    def is_free(self, x, y):
        """
        Check if the cell (x, y) is free.
        """
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
        
        start[0] = start[0] + 0.5*self.resolution
        start[1] = start[1] + 0.5*self.resolution
        target[0] = target[0] + 0.5*self.resolution
        target[1] = target[1] + 0.5*self.resolution

        ax.plot(start[0], start[1], 'go', label='start', markersize=10)
        ax.plot(target[0], target[1], 'ro', label='Goal', markersize=10)
        goal_circle = Circle(target, 1, color='red', alpha=0.3)

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        return fig, ax
    
    def visualize(self, start, target, title= "Environment"):
        self.visualizeWithoutShow(start, target)
        plt.show()

        
def add_polygon_obstacle_to_grid(gridmap, polygon, origin=(0, 0)):
    """
    Rasterize a polygonal obstacle into the gridmap.
    :param gridmap: Instance of GridMap.
    :param polygon: A shapely.geometry.Polygon object.
    :param origin: Origin of world coordinates (for future use).
    """
    #Add your code here
    #gridmap.set_polygonWithoutGrid(polygon.exterior.coords)
    for y in range(0, gridmap.height):
        for x in range(0, gridmap.width):
            p = geo.Point((x + 0.5*gridmap.resolution, y + 0.5*gridmap.resolution))
            if (polygon.contains(p) or polygon.touches(p)):
                gridmap.set_grid(x,y)
                gridmap.set_obstacle(x,y)




grid = GridMap(10, 10, 1)
grid.set_obstacle(1, 1)
print("Is (4, 2) free?", grid.is_free(4, 2))  # False
print("Is (1, 1) free?", grid.is_free(1, 1))  # True
square = geo.Polygon([(2.0, 2.0), (2.5, 2.0), (2.5, 2.5), (2.0, 2.5)])
triangle = geo.Polygon([(1.0, 3.0), (1.5, 4.0), (2.0, 3.0)])
l_shape = geo.Polygon([(3.0, 3.0), (4.0, 3.0), (4.0, 3.5), (3.5, 3.5), (3.5, 4.0), (3.0, 4.0)])

add_polygon_obstacle_to_grid(grid, square)
add_polygon_obstacle_to_grid(grid, triangle)
add_polygon_obstacle_to_grid(grid, l_shape)

grid.visualize([5, 5], [8,8])