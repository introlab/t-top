from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


class StewartCanvas(QWidget):
    def __init__(self):
        super().__init__()

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')

        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_zlabel('Z (m)')

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax.mouse_init()

        self._navigation_toolbar = NavigationToolbar2QT(self._canvas, self)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self._canvas)
        vlayout.addWidget(self._navigation_toolbar)
        self.setLayout(vlayout)

    def draw(self, stewart_state):
        self._ax.cla()

        self._draw_axis()
        self._draw_top(stewart_state.top_state)
        self._draw_bottom(stewart_state.bottom_state)
        self._draw_rods(stewart_state.top_state.get_transformed_anchors(),
                        stewart_state.bottom_state.get_transformed_anchors())

        self._canvas.draw()

    def _draw_axis(self):
        self._ax.plot([0, 0.02], [0, 0], [0, 0], color='red')
        self._ax.plot([0, 0], [0, 0.02], [0, 0], color='green')
        self._ax.plot([0, 0], [0, 0], [0, 0.02], color='blue')

    def _draw_top(self, top_state):
        for shape in top_state.get_transformed_shapes():
            self._ax.plot(shape.x, shape.y, shape.z, color='green')

        anchors = top_state.get_transformed_anchors()
        self._ax.scatter(anchors.x, anchors.y, anchors.z, color='yellow', marker='^')

    def _draw_bottom(self, bottom_state):
        for shape in bottom_state.get_transformed_shapes():
            self._ax.plot(shape.x, shape.y, shape.z, color='red')

        servos = bottom_state.get_transformed_servos()
        self._ax.scatter(servos.x, servos.y, servos.z, color='blue', marker='x')

        horns = bottom_state.get_transformed_horns()
        for horn in horns:
            self._ax.plot(horn.x, horn.y, horn.z, color='orange')

        anchors = bottom_state.get_transformed_anchors()
        self._ax.scatter(anchors.x, anchors.y, anchors.z, color='yellow', marker='^')

    def _draw_rods(self, top_anchors, bottom_anchors):
        for i in range(len(top_anchors.x)):
            x = np.array([top_anchors.x[i], bottom_anchors.x[i]])
            y = np.array([top_anchors.y[i], bottom_anchors.y[i]])
            z = np.array([top_anchors.z[i], bottom_anchors.z[i]])

            self._ax.plot(x, y, z, color='black')
