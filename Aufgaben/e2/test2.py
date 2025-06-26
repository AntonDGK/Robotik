import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

import matplotlib.animation as animation

map = [[0,1000], [0,1000]]
target = (20, 30)
init = (500,500)
obstacles = [[(100,0), (200,0), (200,140), (100,140)],[(600,1000), (600,600), (700, 600), (700, 1000)],[(650,0),(650, 300),(625,300),(625,0)],[(100,850),(100, 750),(200,750),(200,850)]]
target_radius = 20


t_stop = 10  # how many seconds to simulate
history_len = 500  # how many trajectory points to display


def movement(t, state):
    dydx = (1, 1)
    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)



y = np.empty((len(t), 4))
# movement

for i in range(1, len(t)):
    movement(t, i)



fig = plt.subplot(figsize=(8, 8))    
ax = fig.add_subplot(autoscale_on=False, xlim=(map[0]), ylim=(map[1]))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
#    thisx = [0, x1[i], x2[i]]
#    thisy = [0, y1[i], y2[i]]
#
#    history_x = x2[:i]
#    history_y = y2[:i]
#
#    line.set_data(thisx, thisy)
#    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
#    return line, trace, time_text
    return 0, 0, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()