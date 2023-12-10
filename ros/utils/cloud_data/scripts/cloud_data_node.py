#!/usr/bin/env python3

import os
import json

from numpy.lib.function_base import select

import ipinfo
import requests

import rospy
from cloud_data.srv import Location, LocationResponse
from cloud_data.srv import CurrentLocalWeather, CurrentLocalWeatherResponse
from cloud_data.srv import LocalWeatherForecast, LocalWeatherForecastResponse

OPEN_WEATHER_MAP_API_URL = 'https://us-central1-ttop-316419.cloudfunctions.net/weather'


class CloudDataNode:
    def __init__(self):
        self._language = rospy.get_param('~language')
        self._timeout = rospy.get_param('~timeout_s')

        self._location_service = rospy.Service('cloud_data/location', Location, self._handle_location)
        self._current_local_weather_service = rospy.Service('cloud_data/current_local_weather',
                                                            CurrentLocalWeather,
                                                            self._handle_current_local_weather)
        self._local_weather_forecast_service = rospy.Service('cloud_data/local_weather_forecast',
                                                             LocalWeatherForecast,
                                                             self._handle_local_weather_forecast)

    def _handle_location(self, request):
        location = self._get_location()

        return LocationResponse(
            city=location.city,
            region=location.region,
            country_code=location.country,
            country_name=location.country_name
        )

    def _handle_current_local_weather(self, request):
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

        return CurrentLocalWeatherResponse(
            city=location.city,
            region=location.region,
            country_code=location.country,
            country_name=location.country_name,

            temperature_celsius=current_weather['temp'],
            feels_like_temperature_celsius=current_weather['feels_like'],
            pressure_kpa=self._hpa_to_kpa(current_weather['pressure']),
            humidity_percent=current_weather['humidity'],
            clouds_percent=current_weather['clouds'],
            visibility_meter=current_weather['visibility'],
            wind_speed_kph=self._mps_to_kph(current_weather['wind_speed']),
            wind_gust_kph=wind_gust,
            weather_description=weather_description
        )

    def _handle_local_weather_forecast(self, request):
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

        return LocalWeatherForecastResponse(
            city=location.city,
            region=location.region,
            country_code=location.country,
            country_name=location.country_name,

            temperature_morning_celsius=daily_weather['temp']['morn'],
            temperature_day_celsius=daily_weather['temp']['day'],
            temperature_evening_celsius=daily_weather['temp']['eve'],
            temperature_night_celsius=daily_weather['temp']['night'],

            feals_like_temperature_morning_celsius=daily_weather['feels_like']['morn'],
            feals_like_temperature_day_celsius=daily_weather['feels_like']['day'],
            feals_like_temperature_evening_celsius=daily_weather['feels_like']['eve'],
            feals_like_temperature_night_celsius=daily_weather['feels_like']['night'],

            pressure_kpa=self._hpa_to_kpa(daily_weather['pressure']),
            humidity_percent=daily_weather['humidity'],
            clouds_percent=daily_weather['clouds'],
            wind_speed_kph=self._mps_to_kph(daily_weather['wind_speed']),
            wind_gust_kph=wind_gust,
            weather_description=weather_description
        )

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
        rospy.spin()


def main():
    rospy.init_node('cloud_data_node')
    cloud_data_node = CloudDataNode()
    cloud_data_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
