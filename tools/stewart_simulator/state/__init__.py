import numpy as np


class Shape:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'x: ' + str(self.x) + ', y: ' + str(self.y) + ', z: ' + str(self.z)

    def rotate(self, rotation, center=np.array([0, 0, 0])):
        x = np.zeros(len(self.x))
        y = np.zeros(len(self.y))
        z = np.zeros(len(self.z))

        for i in range(len(x)):
            p = np.array([self.x[i], self.y[i], self.z[i]])
            p = rotation.apply(p - center) + center

            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]

        return Shape(x, y, z)

    def translate(self, translation):
        x = np.zeros(len(self.x))
        y = np.zeros(len(self.y))
        z = np.zeros(len(self.z))

        for i in range(len(x)):
            x[i] = self.x[i] + translation[0]
            y[i] = self.y[i] + translation[1]
            z[i] = self.z[i] + translation[2]

        return Shape(x, y, z)
