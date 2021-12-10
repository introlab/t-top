#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

dtheta = 2 * math.pi / 8
movement_beat_duration = 4.0
bpm = 60.0

time_step = 0.1

tf = movement_beat_duration / (bpm / 60.0)
t = np.arange(0, tf / 4, time_step)

theta = dtheta * t / (tf / 4)
theta = np.concatenate((theta, np.flip(theta), -theta, np.flip(-theta)))

t = np.arange(0, tf, time_step)
plt.plot(t, theta, 'o')
plt.show()

for i in range(len(theta)):
    print('[{}],'.format(theta[i]))
