#!/usr/bin/env python

import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv

#Parameters
MICROPHONE_COUNT = 16
MICROPHONE_1_YAW_ANGLE = 0.98174770425
IS_MICROPHONE_1_BOTTOM = False

WALL_ANGLE = 1.4137166941
BOTTOM_RADIUS = 0.155
BOTTOM_MICROPHONE_WALL_HEIGHT = 0.01862439578
TOP_MICROPHONE_WALL_HEIGHT = 0.0599928907
MIDDLE_HEIGHT = (BOTTOM_MICROPHONE_WALL_HEIGHT + TOP_MICROPHONE_WALL_HEIGHT) / 2

Z_OFFSET = -MIDDLE_HEIGHT * math.sin(WALL_ANGLE)


#Position and direction calculation
yaw_angle_step = 2 * math.pi / MICROPHONE_COUNT
bottom_microphone_radius = BOTTOM_RADIUS - BOTTOM_MICROPHONE_WALL_HEIGHT * math.cos(WALL_ANGLE)
top_microphone_radius = BOTTOM_RADIUS - TOP_MICROPHONE_WALL_HEIGHT * math.cos(WALL_ANGLE)
bottom_z = Z_OFFSET + BOTTOM_MICROPHONE_WALL_HEIGHT * math.sin(WALL_ANGLE)
top_z = Z_OFFSET + TOP_MICROPHONE_WALL_HEIGHT * math.sin(WALL_ANGLE)

#Position calculation
microphone_position_x = np.zeros(MICROPHONE_COUNT)
microphone_position_y = np.zeros(MICROPHONE_COUNT)
microphone_position_z = np.zeros(MICROPHONE_COUNT)

current_yaw_angle = MICROPHONE_1_YAW_ANGLE
current_is_bottom = IS_MICROPHONE_1_BOTTOM

for i in range(MICROPHONE_COUNT):
    if current_is_bottom:
        radius = bottom_microphone_radius
        microphone_position_z[i] = bottom_z
    else:
        radius = top_microphone_radius
        microphone_position_z[i] = top_z

    microphone_position_x[i] = radius * math.cos(current_yaw_angle)
    microphone_position_y[i] = radius * math.sin(current_yaw_angle)

    current_is_bottom = not current_is_bottom
    current_yaw_angle += yaw_angle_step

#Direction calculation
microphone_direction_x = np.zeros(MICROPHONE_COUNT)
microphone_direction_y = np.zeros(MICROPHONE_COUNT)
microphone_direction_z = np.zeros(MICROPHONE_COUNT)

current_yaw_angle = MICROPHONE_1_YAW_ANGLE
z = math.tan(math.pi / 2 - WALL_ANGLE)
norm = math.sqrt(1 + z ** 2)

for i in range(MICROPHONE_COUNT):
    x = math.cos(current_yaw_angle)
    y = math.sin(current_yaw_angle)

    microphone_direction_x[i] = x / norm
    microphone_direction_y[i] = y / norm
    microphone_direction_z[i] = z / norm

    current_yaw_angle += yaw_angle_step

print(microphone_direction_x)
print(microphone_direction_y)
print(microphone_direction_z)

#Position and direction drawing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
abs_axix_limit = 1.5 * BOTTOM_RADIUS
ax.set_xlim(-abs_axix_limit, abs_axix_limit)
ax.set_ylim(-abs_axix_limit, abs_axix_limit)
ax.set_zlim(-abs_axix_limit, abs_axix_limit)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.mouse_init()

axis_length = BOTTOM_RADIUS / 2
ax.plot([0, axis_length], [0, 0], [0, 0], color='red')
ax.plot([0, 0], [0, axis_length], [0, 0], color='green')
ax.plot([0, 0], [0, 0], [0, axis_length], color='blue')

direction_length = BOTTOM_RADIUS / 4
for i in range(MICROPHONE_COUNT):
    ax.text(microphone_position_x[i], microphone_position_y[i], microphone_position_z[i], str(i + 1))

    ax.plot([microphone_position_x[i], microphone_position_x[i] + microphone_direction_x[i] * direction_length],
            [microphone_position_y[i], microphone_position_y[i] + microphone_direction_y[i] * direction_length],
            [microphone_position_z[i], microphone_position_z[i] + microphone_direction_z[i] * direction_length], color='orange')

ax.scatter(microphone_position_x, microphone_position_y, microphone_position_z, color='blue', marker='x')

#print configuration
print('')
print('    mics = (')
print('')

for i in range(MICROPHONE_COUNT):
    print('        # Microphone {}'.format(i + 1))
    print('        {')
    print('            mu = ( {}, {}, {} );'.format(microphone_position_x[i],
                                                    microphone_position_y[i],
                                                    microphone_position_z[i]))
    print('            sigma2 = ( +1E-6, 0.0, 0.0, 0.0, +1E-6, 0.0, 0.0, 0.0, +1E-6 );')
    print('            direction = ( {}, {}, {} );'.format(microphone_direction_x[i],
                                                           microphone_direction_y[i],
                                                           microphone_direction_z[i]))
    print('            angle = ( 80.0, 100.0 );')

    if i < MICROPHONE_COUNT - 1:
        print('        },')
    else:
        print('        }')
    print('')

print('    );')

plt.show()
