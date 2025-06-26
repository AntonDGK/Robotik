from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import math
from matplotlib.animation import FFMpegWriter
import time
import pandas as pd
import numpy as np
import matplotlib.animation as ani



# Example Configurations
map = [[0,1000], [0,1000]]
target = (20, 30)
init = (500,500)
obstacles = [[(100,0), (200,0), (200,140), (100,140)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(200,750),(200,850)]]
target_radius = 20

t_stop = 10
dt = 0.1
t = np.arange(0, t_stop, dt)

y = np.empty((len(t), 4))
for i in range(1, len(t)):
    y[i] = y[i - 1] + dt

# TODO: Implement your own visualisation function for the environment, show multiple steps of a changing environment (e.g., a moving robot) as a gif
def configurationSpaces(mapI):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(autoscale_on=False, xlim=(mapI[0][0], mapI[0][1]), ylim=(mapI[1][0], mapI[1][1]))
    ax.set_aspect('equal')
    ax.grid(False)
    return (fig ,ax)

def animate(i):
    return 0, 0, 0

(a, b) = configurationSpaces(map)

ani = ani.FuncAnimation(a, animate, len(y), interval=dt*1000, blit=True)
plt.show()