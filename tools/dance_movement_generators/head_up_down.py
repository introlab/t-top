#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

dz = 0.01
movement_beat_duration = 1.0
bpm = 60.0

time_step = 0.05

tf = movement_beat_duration / (bpm / 60.0)
t = np.arange(0, tf, time_step)
z = dz * np.sin(2 * math.pi / tf * t)

plt.plot(t, z, 'o')
plt.show()

for i in range(len(z)):
    print('[{}, {}, {}, {}, {}, {}, {}],'.format(0, 0, z[i], 0, 0, 0, 1))
