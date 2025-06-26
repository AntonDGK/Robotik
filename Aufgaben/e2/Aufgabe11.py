from random import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import math
from matplotlib.animation import FFMpegWriter
import time
import pandas as pd
import numpy as np
import matplotlib.animation as anime
import matplotlib.animation as animation
from Aufgabe01 import *


map = [[0,1000], [0,1000]]

def animate(i):
    return 0, 0, 0


(a, b) = configurationSpaces(map)
ani = animation.FuncAnimation(a, animate, len(y), interval=dt*1000, blit=True)
plt.show()