import math

import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QWidget, QTabWidget, QGroupBox, QSlider, QLabel, QDoubleSpinBox, QVBoxLayout, \
    QHBoxLayout, QFormLayout, QPushButton
from scipy.spatial.transform import Rotation

from ui.controller_parameters_dialog import ControllerParametersDialog
from ui.stewart_canvas import StewartCanvas

SERVO_COUNT = 6

POSITION_RELATIVE_RANGE_MIN = -0.03
POSITION_RELATIVE_RANGE_MAX = 0.03
POSITION_STEP = 0.001
POSITION_DECIMALS = 3

ORIENTATION_ANGLE_MIN = -math.pi / 4
ORIENTATION_ANGLE_MAX = math.pi / 4

SERVO_ANGLE_SLIDER_SCALE = 10


class StewartSimulatorInterface(QWidget):
    def __init__(self, configuration, stewart_state):
        super().__init__()
        self._configuration = configuration
        self._stewart_state = stewart_state

        self.setWindowTitle('Stewart Simulator')

        self._stewart_canvas = StewartCanvas()
        self._tab_widget = QTabWidget()
        self._position_status_label = QLabel('')

        self._show_controller_parameters_button = QPushButton('Show Controller Parameters')
        self._show_controller_parameters_button.clicked.connect(self._on_show_controller_parameters_button_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self._stewart_canvas)
        layout.addWidget(self._tab_widget)
        layout.addWidget(self._position_status_label)
        layout.addWidget(self._show_controller_parameters_button)
        self.setLayout(layout)

        self._forward_kinematics_widget = self._create_forward_kinematics_widget()
        self._inverse_kinematics_widget = self._create_inverse_kinematics_widget()
        self._state_widget = self._create_state_widget()

        self._tab_widget.addTab(self._forward_kinematics_widget, 'Forward Kinematics')
        self._tab_widget.addTab(self._inverse_kinematics_widget, 'Inverse Kinematics')
        self._tab_widget.addTab(self._state_widget, 'State')

        self._stewart_canvas.draw(self._stewart_state)
        self._update_displayed_state()
        self._set_valid_position_status()

    def _create_forward_kinematics_widget(self):
        widget = QWidget()

        self._forward_kinematics_servo_labels = [QLabel('0') for i in range(SERVO_COUNT)]
        self._forward_kinematics_servo_sliders = self._create_forward_kinematics_servo_sliders()

        vlayout = QVBoxLayout()

        for i in range(SERVO_COUNT):
            hlayout = QHBoxLayout()
            hlayout.addWidget(QLabel('Servo Angle ' + str(i + 1) + ' (°) :'))
            hlayout.addWidget(self._forward_kinematics_servo_sliders[i])
            hlayout.addWidget(self._forward_kinematics_servo_labels[i])

            vlayout.addLayout(hlayout)

        widget.setLayout(vlayout)
        return widget

    def _create_forward_kinematics_servo_sliders(self):
        sliders = []

        for i in range(SERVO_COUNT):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(math.degrees(self._configuration.servo_angle_min) * SERVO_ANGLE_SLIDER_SCALE))
            slider.setMaximum(int(math.degrees(self._configuration.servo_angle_max) * SERVO_ANGLE_SLIDER_SCALE))
            slider.setValue(0)
            slider.valueChanged.connect(self._on_forward_kinematics_servo_slider_value_changed)
            sliders.append(slider)

        return sliders

    def _create_inverse_kinematics_widget(self):
        widget = QWidget()

        self._inverse_kinematics_position_x_spin_box = self._create_inverse_kinematics_position_spin_box()
        self._inverse_kinematics_position_y_spin_box = self._create_inverse_kinematics_position_spin_box()
        self._inverse_kinematics_position_z_spin_box = self._create_inverse_kinematics_position_spin_box(
            self._stewart_state.top_state.get_initial_z())

        self._inverse_kinematics_orientation_x_spin_box = self._create_inverse_kinematics_orientation_spin_box()
        self._inverse_kinematics_orientation_y_spin_box = self._create_inverse_kinematics_orientation_spin_box()
        self._inverse_kinematics_orientation_z_spin_box = self._create_inverse_kinematics_orientation_spin_box()
        self._inverse_kinematics_orientation_angle_spin_box = self._create_inverse_kinematics_orientation_spin_box()
        self._inverse_kinematics_orientation_angle_spin_box.setRange(math.degrees(ORIENTATION_ANGLE_MIN),
                                                                     math.degrees(ORIENTATION_ANGLE_MAX))

        position_group_box = QGroupBox('Position')
        orientation_group_box = QGroupBox('Orientation')

        layout = QVBoxLayout()
        layout.addWidget(position_group_box)
        layout.addWidget(orientation_group_box)
        widget.setLayout(layout)

        form_layout = QFormLayout()
        position_group_box.setLayout(form_layout)

        form_layout.addRow('X (m) :', self._inverse_kinematics_position_x_spin_box)
        form_layout.addRow('Y (m) :', self._inverse_kinematics_position_y_spin_box)
        form_layout.addRow('Z (m) :', self._inverse_kinematics_position_z_spin_box)

        form_layout = QFormLayout()
        orientation_group_box.setLayout(form_layout)

        form_layout.addRow('X :', self._inverse_kinematics_orientation_x_spin_box)
        form_layout.addRow('Y :', self._inverse_kinematics_orientation_y_spin_box)
        form_layout.addRow('Z :', self._inverse_kinematics_orientation_z_spin_box)
        form_layout.addRow('θ (°) :', self._inverse_kinematics_orientation_angle_spin_box)

        return widget

    def _create_inverse_kinematics_position_spin_box(self, center=0.0):
        spin_box = QDoubleSpinBox()
        spin_box.setRange(POSITION_RELATIVE_RANGE_MIN + center, POSITION_RELATIVE_RANGE_MAX + center)
        spin_box.setValue(center)
        spin_box.setSingleStep(POSITION_STEP)
        spin_box.setDecimals(POSITION_DECIMALS)

        spin_box.valueChanged.connect(self._on_inverse_kinematics_spin_box_value_changed)
        return spin_box

    def _create_inverse_kinematics_orientation_spin_box(self):
        spin_box = QDoubleSpinBox()
        spin_box.valueChanged.connect(self._on_inverse_kinematics_spin_box_value_changed)
        return spin_box

    def _create_state_widget(self):
        widget = QWidget()
        h_layout = QHBoxLayout()
        form_layout = QFormLayout()
        h_layout.addLayout(form_layout)

        self._state_servo_angle_labels = []
        for i in range(SERVO_COUNT):
            self._state_servo_angle_labels.append(QLabel('0'))
            form_layout.addRow('Servo Angle ' + str(i + 1) + ' (°) :', self._state_servo_angle_labels[i])

        self._state_position_x_label = QLabel('0')
        self._state_position_y_label = QLabel('0')
        self._state_position_z_label = QLabel('0')

        self._state_orientation_x_label = QLabel('0')
        self._state_orientation_y_label = QLabel('0')
        self._state_orientation_z_label = QLabel('0')
        self._state_orientation_angle_label = QLabel('0')

        form_layout.addRow('Position X (m) :', self._state_position_x_label)
        form_layout.addRow('Position Y (m) :', self._state_position_y_label)
        form_layout.addRow('Position Z (m) :', self._state_position_z_label)

        form_layout.addRow('Orientation X :', self._state_orientation_x_label)
        form_layout.addRow('Orientation Y :', self._state_orientation_y_label)
        form_layout.addRow('Orientation Z :', self._state_orientation_z_label)
        form_layout.addRow('Orientation θ (°) :', self._state_orientation_angle_label)

        form_layout = QFormLayout()
        h_layout.addLayout(form_layout)

        self._state_top_ball_joint_angle_labels = []
        self._state_bottom_ball_joint_angle_labels = []
        for i in range(SERVO_COUNT):
            self._state_top_ball_joint_angle_labels.append(QLabel('0'))
            form_layout.addRow('Bottom Ball Joint Angle ' + str(i + 1) + ' (°) :',
                               self._state_top_ball_joint_angle_labels[i])

            self._state_bottom_ball_joint_angle_labels.append(QLabel('0'))
            form_layout.addRow('Top Ball Joint Angle ' + str(i + 1) + ' (°) :',
                               self._state_bottom_ball_joint_angle_labels[i])

        widget.setLayout(h_layout)

        return widget

    def _update_displayed_state(self):
        servo_angles = self._stewart_state.bottom_state.get_servo_angles()
        for i in range(SERVO_COUNT):
            self._state_servo_angle_labels[i].setText(str(math.degrees(servo_angles[i])))

        position = self._stewart_state.top_state.get_position()
        orientation = self._stewart_state.top_state.get_orientation().as_rotvec()
        angle = np.linalg.norm(orientation)
        if angle > 0:
            orientation /= angle

        self._state_position_x_label.setText(str(position[0]))
        self._state_position_y_label.setText(str(position[1]))
        self._state_position_z_label.setText(str(position[2]))

        self._state_orientation_x_label.setText(str(orientation[0]))
        self._state_orientation_y_label.setText(str(orientation[1]))
        self._state_orientation_z_label.setText(str(orientation[2]))
        self._state_orientation_angle_label.setText(str(math.degrees(angle)))

        top_ball_joint_angles, bottom_ball_joint_angles = self._stewart_state.get_ball_joint_angles()
        for i in range(SERVO_COUNT):
            self._state_top_ball_joint_angle_labels[i].setText(str(math.degrees(top_ball_joint_angles[i])))
            self._state_bottom_ball_joint_angle_labels[i].setText(str(math.degrees(bottom_ball_joint_angles[i])))

    def _set_valid_position_status(self):
        self._position_status_label.setStyleSheet('QLabel { background-color : green; }')

    def _set_invalid_position_status(self):
        self._position_status_label.setStyleSheet('QLabel { background-color : red; }')

    def _on_forward_kinematics_servo_slider_value_changed(self, value):
        i = self._forward_kinematics_servo_sliders.index(self.sender())
        self._forward_kinematics_servo_labels[i].setText(str(value / SERVO_ANGLE_SLIDER_SCALE))

        servo_angles = [math.radians(slider.value() / SERVO_ANGLE_SLIDER_SCALE)
                        for slider in self._forward_kinematics_servo_sliders]

        try:
            self._stewart_state.set_servo_angles(servo_angles)
            self._set_valid_position_status()
        except:
            self._set_invalid_position_status()

        self._stewart_canvas.draw(self._stewart_state)
        self._update_displayed_state()

    def _on_inverse_kinematics_spin_box_value_changed(self):
        position = self._get_inverse_kinematics_position_from_ui()
        orientation = self._get_inverse_kinematics_orientation_from_ui()

        try:
            self._stewart_state.set_top_pose(position, orientation)
            self._set_valid_position_status()
        except:
            self._set_invalid_position_status()
        self._stewart_canvas.draw(self._stewart_state)
        self._update_displayed_state()

    def _get_inverse_kinematics_position_from_ui(self):
        return np.array([self._inverse_kinematics_position_x_spin_box.value(),
                         self._inverse_kinematics_position_y_spin_box.value(),
                         self._inverse_kinematics_position_z_spin_box.value()])

    def _get_inverse_kinematics_orientation_from_ui(self):
        rotvec = np.array([self._inverse_kinematics_orientation_x_spin_box.value(),
                           self._inverse_kinematics_orientation_y_spin_box.value(),
                           self._inverse_kinematics_orientation_z_spin_box.value()])
        angle = math.radians(self._inverse_kinematics_orientation_angle_spin_box.value())
        rotvec_norm = np.linalg.norm(rotvec)
        if rotvec_norm > 0:
            rotvec /= rotvec_norm
            orientation = Rotation.from_rotvec(rotvec * angle)
        else:
            orientation = Rotation.from_euler('xyz', [0, 0, 0])

        return orientation

    def _on_show_controller_parameters_button_clicked(self):
        dialog = ControllerParametersDialog(self, self._stewart_state)
        dialog.setModal(True)
        dialog.show()
