#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

dtheta = 0.174533
movement_beat_duration = 2.0
bpm = 60.0

time_step = 0.05

tf = movement_beat_duration / (bpm / 60.0)
t = np.arange(0, tf, time_step)
theta = dtheta * np.sin(2 * math.pi / tf * t)

plt.plot(t, theta, 'o')
plt.show()

for i in range(len(theta)):
    q = R.from_rotvec(np.array([theta[i], 0, 0])).as_quat()
    print('[{}, {}, {}, {}, {}, {}, {}],'.format(0, 0, 0, q[0], q[1], q[2], q[3]))
