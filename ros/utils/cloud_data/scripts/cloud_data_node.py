#!/usr/bin/env python3

import os
import json

import ipinfo
import requests

import rclpy
import rclpy.node

from cloud_data.srv import Location, CurrentLocalWeather, LocalWeatherForecast

OPEN_WEATHER_MAP_API_URL = 'https://us-central1-ttop-316419.cloudfunctions.net/weather'


class CloudDataNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('cloud_data_node')

        self._language = self.declare_parameter('language', 'en').get_parameter_value().string_value
        self._timeout = self.declare_parameter('timeout_s', 10.0).get_parameter_value().double_value

        self._location_service = self.create_service(Location, 'cloud_data/location', self._handle_location)
        self._current_local_weather_service = self.create_service(CurrentLocalWeather,
                                                                  'cloud_data/current_local_weather',
                                                                  self._handle_current_local_weather)
        self._local_weather_forecast_service = self.create_service(LocalWeatherForecast,
                                                                   'cloud_data/local_weather_forecast',
                                                                   self._handle_local_weather_forecast)

    def _handle_location(self, request, response):
        try:
            location = self._get_location()

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name
        except Exception as e:
            self.get_logger().error(f'An error occured while retrieving the location: {e}')
            response.ok = False

        return response

    def _handle_current_local_weather(self, request, response):
        try:
            location = self._get_location()
            weather = self._get_weather(location.latitude, location.longitude)
            current_weather = weather['current']

            if 'wind_gust' in current_weather:
                wind_gust = self._mps_to_kph(current_weather['wind_gust'])
            else:
                wind_gust = -1.0

            if len(current_weather['weather']) == 0:
                weather_description = ''
            else:
                weather_description = current_weather['weather'][0]['description']

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_celsius = current_weather['temp']
            response.feels_like_temperature_celsius = current_weather['feels_like']
            response.pressure_kpa = self._hpa_to_kpa(current_weather['pressure'])
            response.humidity_percent = float(current_weather['humidity'])
            response.clouds_percent = float(current_weather['clouds'])
            response.visibility_meter = float(current_weather['visibility'])
            response.wind_speed_kph = self._mps_to_kph(current_weather['wind_speed'])
            response.wind_gust_kph = wind_gust
            response.weather_description = weather_description
        except Exception as e:
            self.get_logger().error(f'An error occured while retrieving the local weather: {e}')
            response.ok = False

        return response

    def _handle_local_weather_forecast(self, request, response):
        try:
            location = self._get_location()
            weather = self._get_weather(location.latitude, location.longitude)

            if request.relative_day >= len(weather['daily']):
                raise ValueError('Invalid relative day')

            daily_weather = weather['daily'][request.relative_day]

            if 'wind_gust' in daily_weather:
                wind_gust = self._mps_to_kph(daily_weather['wind_gust'])
            else:
                wind_gust = -1.0

            if len(daily_weather['weather']) == 0:
                weather_description = ''
            else:
                weather_description = daily_weather['weather'][0]['description']

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_morning_celsius = daily_weather['temp']['morn']
            response.temperature_day_celsius = daily_weather['temp']['day']
            response.temperature_evening_celsius = daily_weather['temp']['eve']
            response.temperature_night_celsius = daily_weather['temp']['night']

            response.feals_like_temperature_morning_celsius = daily_weather['feels_like']['morn']
            response.feals_like_temperature_day_celsius = daily_weather['feels_like']['day']
            response.feals_like_temperature_evening_celsius = daily_weather['feels_like']['eve']
            response.feals_like_temperature_night_celsius = daily_weather['feels_like']['night']

            response.pressure_kpa = self._hpa_to_kpa(daily_weather['pressure'])
            response.humidity_percent = float(daily_weather['humidity'])
            response.clouds_percent = float(daily_weather['clouds'])
            response.wind_speed_kph = self._mps_to_kph(daily_weather['wind_speed'])
            response.wind_gust_kph = wind_gust
            response.weather_description = weather_description
        except Exception as e:
            self.get_logger().error(f'An error occured while retrieving the local weather forecast: {e}')
            response.ok = False

        return response

    def _get_location(self):
        handler = ipinfo.getHandler()
        return handler.getDetails()

    def _get_weather(self, latitude, longitude):
        params = {
            'lat': str(latitude),
            'lon': str(longitude),
            'exclude': 'minutely,hourly,alerts',
            'units': 'metric',
            'lang': self._language,
            'appid': os.environ.get('OPEN_WEATHER_MAP_API_KEY')
        }
        response = requests.get(OPEN_WEATHER_MAP_API_URL, params=params, timeout=self._timeout)
        return json.loads(response.text)

    def _hpa_to_kpa(self, hpa):
        return hpa / 10

    def _mps_to_kph(self, mps):
        return mps * 3.6

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    cloud_data_node = CloudDataNode()

    try:
        cloud_data_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        cloud_data_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
