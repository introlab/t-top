#!/usr/bin/env python3

import rospy
import psutil
import os
import re
import json
from opentera_webrtc_ros_msgs.msg import RobotStatus
from std_msgs.msg import String, Float32MultiArray
from subprocess import Popen, PIPE
from typing import List, Optional, Union
from threading import Lock


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


class BaseStatus:
    def __init__(self,
                 percentage: Optional[float] = None,
                 voltage: Optional[float] = None,
                 current: Optional[float] = None,
                 is_plugged_in: Optional[Union[bool, float]] = None,
                 is_charging: Optional[Union[bool, float]] = None) -> None:
        self.percentage = clamp(percentage or 0.0, 0, 100)
        self.voltage = voltage or 0.0
        self.current = current or 0.0
        self.is_plugged_in = bool(
            is_plugged_in) if is_plugged_in is not None else False
        self.is_charging = bool(
            is_charging) if is_charging is not None else False


def get_command_output(cmd: List[str]) -> str:
    p = Popen(cmd, stdout=PIPE)
    return p.communicate()[0].decode('utf-8')


class RobotStatusPublisher():
    def __init__(self):
        rospy.init_node("robot_status_publisher")

        self.base_status = BaseStatus()
        self.base_status_lock = Lock()
        self.pub_rate = 1

        self.status_pub = rospy.Publisher(
            '/robot_status', RobotStatus, queue_size=10)
        self.status_webrtc_pub = rospy.Publisher(
            '/webrtc_data_outgoing', String, queue_size=10)

        self.base_status_sub = rospy.Subscriber(
            "/opencr/base_status",
            Float32MultiArray, self.base_status_cb, queue_size=1)

    def get_ip_address(self, ifname: str):
        try:
            address = get_command_output(["ip", "addr", "show", ifname]).split(
                "inet ")[1].split("/")[0]
        except Exception:
            address = '127.0.0.1'

        return address

    def get_disk_usage(self, mount_point='/'):
        result = os.statvfs(mount_point)
        total_blocks = result.f_blocks
        free_blocks = result.f_bfree
        return 100 - (free_blocks * 100 / total_blocks)

    def base_status_cb(self, msg):
        with self.base_status_lock:
            self.base_status = BaseStatus(*msg.data)

    def run(self):
        rate = rospy.Rate(self.pub_rate)
        while not rospy.is_shutdown():
            # Fill timestamp
            status = RobotStatus()
            status.header.stamp = rospy.Time.now()

            # Fill robot info
            with self.base_status_lock:
                status.battery_voltage = self.base_status.voltage
                status.battery_current = self.base_status.current
                status.battery_level = self.base_status.percentage
                status.is_charging = self.base_status.is_plugged_in

            status.cpu_usage = psutil.cpu_percent()
            status.mem_usage = psutil.virtual_memory().percent
            status.disk_usage = self.get_disk_usage()

            status.wifi_network = get_command_output(["iwgetid"])
            if status.wifi_network:
                wifi_interface_name = status.wifi_network.split()[0]

                wifi_usage = get_command_output(
                    ["iwconfig", wifi_interface_name])
                wifi_strength_re = re.search(
                    r'Link Quality=(\d+)/(\d+)', wifi_usage)
                if wifi_strength_re is not None and len(wifi_strength_re.groups()) == 2:
                    numerator = int(wifi_strength_re.group(1))
                    denominator = int(wifi_strength_re.group(2))
                else:
                    numerator, denominator = (0, 1)
                status.wifi_strength = numerator / denominator * 100
                status.local_ip = self.get_ip_address(wifi_interface_name)
            else:
                status.wifi_strength = 0
                status.local_ip = '127.0.0.1'

            # TODO Get data about mute and camera status from webrtc when it is implemented
            status.is_muted = False
            status.is_camera_on = True

            # Publish for webrtc
            status_dict = {
                'type': 'robotStatus',
                'timestamp': status.header.stamp.secs,
                'status': {
                    'isCharging': status.is_charging,
                    'batteryVoltage': status.battery_voltage,
                    'batteryCurrent': status.battery_current,
                    'batteryLevel': status.battery_level,
                    'cpuUsage': status.cpu_usage,
                    'memUsage': status.mem_usage,
                    'diskUsage': status.disk_usage,
                    'wifiNetwork': status.wifi_network,
                    'wifiStrength': status.wifi_strength,
                    'localIp': status.local_ip,
                    'isMuted': status.is_muted,
                    'isCameraOn': status.is_camera_on
                }
            }
            self.status_webrtc_pub.publish(json.dumps(status_dict))

            # Publish
            self.status_pub.publish(status)

            rate.sleep()


if __name__ == '__main__':
    print("Robot Status Publisher Starting")
    try:
        robot_status = RobotStatusPublisher()
        robot_status.run()
    except rospy.ROSInterruptException:
        pass
