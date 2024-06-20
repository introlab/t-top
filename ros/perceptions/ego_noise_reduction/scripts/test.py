#!/usr/bin/env python3

import threading
import math
import time

import rclpy
import rclpy.node

from hbba_lite_srvs.srv import SetOnOffFilterState

from t_top import MovementCommands


PAUSE_DURATION_S = 5.0


class EgoNoiseReductionTestNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('ego_noise_reduction_test_node')
        self._movement_commands = MovementCommands(self)

    def run(self):
        self._enable_on_off_filter('ego_noise_reduction/filter_state')
        self._enable_on_off_filter('pose/filter_state')

        executer_thread = threading.Thread(target=lambda: rclpy.spin(self))
        executer_thread.start()

        while rclpy.ok():
            time.sleep(PAUSE_DURATION_S)
            self._movement_commands.move_torso(math.pi / 2, should_wait=True)
            self._movement_commands.move_torso(-math.pi / 2, should_wait=True)

            self._movement_commands.move_yes(speed_rad_sec=1.0)
            self._movement_commands.move_no(speed_rad_sec=0.5)
            self._movement_commands.move_maybe(speed_rad_sec=1.5)

        executer_thread.join()

    def _enable_on_off_filter(self, name):
        client = self.create_client(SetOnOffFilterState, name)
        client.wait_for_service()

        request = SetOnOffFilterState.Request()
        request.is_filtering_all_messages = False

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)


def main():
    rclpy.init()
    ego_noise_reduction_test_node = EgoNoiseReductionTestNode()

    try:
        ego_noise_reduction_test_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        ego_noise_reduction_test_node.destroy_node()
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
