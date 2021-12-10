#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

radius = 0.01
movement_beat_duration = 4.0
bpm = 60.0

time_step = 0.05

tf = movement_beat_duration / (bpm / 60.0)
t = np.arange(0, tf, time_step)
x = radius * np.cos(2 * math.pi / tf * t)
y = radius * np.sin(2 * math.pi / tf * t)

plt.plot(t, x, 'o')
plt.show()

plt.plot(x, y, 'o')
plt.show()

for i in range(len(x)):
    print('[{}, {}, {}, {}, {}, {}, {}],'.format(x[i], y[i], 0, 0, 0, 0, 1))
