#!/usr/bin/env python3

import rospy
import psutil
import os
import re
import json
from opentera_webrtc_ros_msgs.msg import RobotStatus
from std_msgs.msg import String, Float32, Bool
from daemon_ros_client.msg import BaseStatus
from subprocess import Popen, PIPE
from typing import List, Optional, Union
from threading import Lock


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


class BaseStatusData:
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

        self.pub_rate = 1
        self.status_pub = rospy.Publisher(
            '/robot_status', RobotStatus, queue_size=10)
        self.status_webrtc_pub = rospy.Publisher(
            '/webrtc_data_outgoing', String, queue_size=10)

        self.mic_volume_sub = rospy.Subscriber(
            'mic_volume', Float32, self._set_mic_volume_cb, queue_size=10)
        self.mic_volume = 1
        self.enable_camera_sub = rospy.Subscriber(
            'enable_camera', Bool, self._set_enable_camera_cb, queue_size=10)
        self.camera_enabled = True
        self.volume_sub = rospy.Subscriber(
            'volume', Float32, self._set_volume_cb, queue_size=10)
        self.volume = 1
        self.io = psutil.net_io_counters(pernic=True)
        self.bytes_sent = 0
        self.bytes_recv = 0

        self.base_status = BaseStatusData()
        self.base_status_lock = Lock()
        self.base_status_sub = rospy.Subscriber(
            "daemon/base_status",
            BaseStatus, self._base_status_cb, queue_size=1)

    def get_ip_address(self, ifname: str) -> str:
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

    def _set_mic_volume_cb(self, msg):
        self.mic_volume: float = msg.data

    def _set_enable_camera_cb(self, msg):
        self.camera_enabled: bool = msg.data

    def _set_volume_cb(self, msg):
        self.volume: float = msg.data

    def _base_status_cb(self, msg):
        with self.base_status_lock:
            self.base_status = BaseStatusData(
                msg.state_of_charge,
                msg.voltage,
                msg.current,
                msg.is_psu_connected,
                msg.is_battery_charging
            )

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

            status.mic_volume = self.mic_volume
            status.is_camera_on = self.camera_enabled
            status.volume = self.volume

            status.wifi_network = get_command_output(["iwgetid"])
            if status.wifi_network != "":
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

                io_2 = psutil.net_io_counters(pernic=True)
                status.upload_speed = (io_2[wifi_interface_name].bytes_sent - self.bytes_sent) * 8
                status.download_speed = (io_2[wifi_interface_name].bytes_recv - self.bytes_recv) * 8
                self.bytes_sent = io_2[wifi_interface_name].bytes_sent
                self.bytes_recv = io_2[wifi_interface_name].bytes_recv
            else:
                status.wifi_strength = 0
                status.local_ip = '127.0.0.1'
                status.upload_speed = 0
                status.download_speed = 0

            # Publish for ROS
            self.status_pub.publish(status)

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
                    'uploadSpeed': status.upload_speed,
                    'downloadSpeed': status.download_speed,
                    'localIp': status.local_ip,
                    'micVolume': status.mic_volume,
                    'isCameraOn': status.is_camera_on,
                    'volume': status.volume,
                }
            }
            self.status_webrtc_pub.publish(json.dumps(status_dict))

            if rospy.is_shutdown():
                break

            rate.sleep()


if __name__ == '__main__':
    print("Robot Status Publisher Starting")
    try:
        robot_status = RobotStatusPublisher()
        robot_status.run()
    except rospy.ROSInterruptException:
        pass
