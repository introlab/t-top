from PySide2.QtWidgets import QDialog, QLabel, QTextEdit, QVBoxLayout

from controller_parameters.opencr import generate_kinematics_controller_parameters_code


class ControllerParametersDialog(QDialog):
    def __init__(self, parent, stewart_state):
        super().__init__(parent)

        self.setWindowTitle('Stewart Simulator - Controller Parameters')

        kinematics_parameters_text = generate_kinematics_controller_parameters_code(
            stewart_state.get_kinematics_controller_parameters())

        self._kinematics_parameters_text_edit = QTextEdit()
        self._kinematics_parameters_text_edit.setReadOnly(True)
        self._kinematics_parameters_text_edit.setText(kinematics_parameters_text)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Controller Parameters :'))
        layout.addWidget(self._kinematics_parameters_text_edit)
        self.setLayout(layout)
