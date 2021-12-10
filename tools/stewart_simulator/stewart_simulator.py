import os
import sys

from PySide2.QtWidgets import QApplication

from configuration.global_configuration import GlobalConfiguration
from state.stewart_state import StewartState
from ui.stewart_simulator_interface import StewartSimulatorInterface


def main():
    configuration = GlobalConfiguration(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configuration.json'))
    stewart_state = StewartState(configuration)

    app = QApplication(sys.argv)
    interface = StewartSimulatorInterface(configuration, stewart_state)
    interface.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
