#!/usr/bin/env python3

import json
import rclpy
import threading
from rclpy.node import Node
from behavior_srvs.srv import ChatToolsFunctionCall
from cloud_data.srv import CurrentLocalWeather2

class WeatherToolBridge(Node):
    def __init__(self):
        super().__init__('weather_tool_bridge')

        self.srv = self.create_service(
            ChatToolsFunctionCall,
            '/chat/tools/functions/get_current_weather',
            self.handle_tool_call
        )

        self._weather_client = self.create_client(
            CurrentLocalWeather2,
            '/cloud_data/current_local_weather'
        )

        self.get_logger().info('Weather tool bridge ready.')

    def handle_tool_call(self, request, response):
        self.get_logger().info(f"Tool call for weather received: {request.function_name}")

        # Call cloud_data service
        if not self._weather_client.wait_for_service(timeout_sec=5.0):
            response.ok = False
            response.result = json.dumps({'error': 'Weather service unavailable'})
            return response

        weather_req = CurrentLocalWeather2.Request()
        future = self._weather_client.call_async(weather_req)

        # Use an event to wait non-blockingly
        done_event = threading.Event()

        def _on_complete(fut):
            done_event.set()

        future.add_done_callback(_on_complete)

        self.get_logger().info("Waiting for weather service...")
        if not done_event.wait(timeout=5.0):
            self.get_logger().error("Timeout waiting for weather response")
            response.ok = False
            response.result = json.dumps({'error': 'Timeout waiting for weather'})
            return response

        self.get_logger().info("Weather service call finished")

        if future.result() is not None and future.result().ok:
            res = future.result()
            payload = {
                'city': res.city,
                'region': res.region,
                'country': res.country_name,
                'temperature_celsius': res.temperature_celsius,
                'wind_speed_kph': res.wind_speed_kph
            }
            response.ok = True
            response.result = json.dumps(payload)
        else:
            response.ok = False
            response.result = json.dumps({'error': 'Failed to get weather'})

        return response

def main():
    rclpy.init()
    node = WeatherToolBridge()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
