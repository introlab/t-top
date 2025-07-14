#!/usr/bin/env python3

import os
import json

import ipinfo
import requests

import rclpy
import rclpy.node
from rclpy.executors import MultiThreadedExecutor


from datetime import datetime

from cloud_data.srv import (
    Location,
    CurrentLocalWeather,
    LocalWeatherForecast,
    CurrentLocalWeatherOpenMeteo,
    LocalWeatherForecastOpenMeteo,
)

OPEN_WEATHER_MAP_API_URL = "https://us-central1-ttop-316419.cloudfunctions.net/weather"


class OpenWeatherCloudDataNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cloud_data_open_weather_node")

        self._language = (
            self.declare_parameter("language", "en").get_parameter_value().string_value
        )
        self._timeout = (
            self.declare_parameter("timeout_s", 10.0).get_parameter_value().double_value
        )

        self._location_service = self.create_service(
            Location, "cloud_data/open_weather/location", self._handle_location
        )
        self._current_local_weather_service = self.create_service(
            CurrentLocalWeather,
            "cloud_data/open_weather/current_local_weather",
            self._handle_current_local_weather,
        )
        self._local_weather_forecast_service = self.create_service(
            LocalWeatherForecast,
            "cloud_data/open_weather/local_weather_forecast",
            self._handle_local_weather_forecast,
        )

    def _handle_location(self, request, response):
        try:
            location = self._get_location()

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name
        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the location: {e}"
            )
            response.ok = False

        return response

    def _handle_current_local_weather(self, request, response):
        self.get_logger().info(f"Fetching current_local_weather")
        try:
            location = self._get_location()
            weather = self._get_weather(location.latitude, location.longitude)
            current_weather = weather["current"]

            if "wind_gust" in current_weather:
                wind_gust = self._mps_to_kph(current_weather["wind_gust"])
            else:
                wind_gust = -1.0

            if len(current_weather["weather"]) == 0:
                weather_description = ""
            else:
                weather_description = current_weather["weather"][0]["description"]

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_celsius = current_weather["temp"]
            response.feels_like_temperature_celsius = current_weather["feels_like"]
            response.pressure_kpa = self._hpa_to_kpa(current_weather["pressure"])
            response.humidity_percent = float(current_weather["humidity"])
            response.clouds_percent = float(current_weather["clouds"])
            response.visibility_meter = float(current_weather["visibility"])
            response.wind_speed_kph = self._mps_to_kph(current_weather["wind_speed"])
            response.wind_gust_kph = wind_gust
            response.weather_description = weather_description
        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the local weather: {e}"
            )
            response.ok = False

        self.get_logger().info(f"Returning current_local_weather, ok: {response.ok}")
        return response

    def _handle_local_weather_forecast(self, request, response):
        self.get_logger().info(f"Fetching local_weather_forecast")
        try:
            location = self._get_location()
            weather = self._get_weather(location.latitude, location.longitude)

            if request.relative_day >= len(weather["daily"]):
                raise ValueError("Invalid relative day")

            daily_weather = weather["daily"][request.relative_day]

            if "wind_gust" in daily_weather:
                wind_gust = self._mps_to_kph(daily_weather["wind_gust"])
            else:
                wind_gust = -1.0

            if len(daily_weather["weather"]) == 0:
                weather_description = ""
            else:
                weather_description = daily_weather["weather"][0]["description"]

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_morning_celsius = daily_weather["temp"]["morn"]
            response.temperature_day_celsius = daily_weather["temp"]["day"]
            response.temperature_evening_celsius = daily_weather["temp"]["eve"]
            response.temperature_night_celsius = daily_weather["temp"]["night"]

            response.feals_like_temperature_morning_celsius = daily_weather[
                "feels_like"
            ]["morn"]
            response.feals_like_temperature_day_celsius = daily_weather["feels_like"][
                "day"
            ]
            response.feals_like_temperature_evening_celsius = daily_weather[
                "feels_like"
            ]["eve"]
            response.feals_like_temperature_night_celsius = daily_weather["feels_like"][
                "night"
            ]

            response.pressure_kpa = self._hpa_to_kpa(daily_weather["pressure"])
            response.humidity_percent = float(daily_weather["humidity"])
            response.clouds_percent = float(daily_weather["clouds"])
            response.wind_speed_kph = self._mps_to_kph(daily_weather["wind_speed"])
            response.wind_gust_kph = wind_gust
            response.weather_description = weather_description
        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the local weather forecast: {e}"
            )
            response.ok = False

        self.get_logger().info(f"Returning local_weather_forecast, ok: {response.ok}")
        return response

    def _get_location(self):
        handler = ipinfo.getHandler()
        return handler.getDetails()

    def _get_weather(self, latitude, longitude):
        params = {
            "lat": str(latitude),
            "lon": str(longitude),
            "exclude": "minutely,hourly,alerts",
            "units": "metric",
            "lang": self._language,
            "appid": os.environ.get("OPEN_WEATHER_MAP_API_KEY"),
        }
        response = requests.get(
            OPEN_WEATHER_MAP_API_URL, params=params, timeout=self._timeout
        )
        return json.loads(response.text)

    def _hpa_to_kpa(self, hpa):
        return hpa / 10

    def _mps_to_kph(self, mps):
        return mps * 3.6

    def run(self):
        rclpy.spin(self)


OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoCloudDataNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cloud_data_open_meteo_node")

        self._language = (
            self.declare_parameter("language", "en").get_parameter_value().string_value
        )
        self._timeout = (
            self.declare_parameter("timeout_s", 10.0).get_parameter_value().double_value
        )

        self._location_service = self.create_service(
            Location, "cloud_data/open_meteo/location", self._handle_location
        )
        self._current_local_weather_service = self.create_service(
            CurrentLocalWeatherOpenMeteo,
            "cloud_data/open_meteo/current_local_weather",
            self._handle_current_local_weather,
        )
        self._local_weather_forecast_service = self.create_service(
            LocalWeatherForecastOpenMeteo,
            "cloud_data/open_meteo/local_weather_forecast",
            self._handle_local_weather_forecast,
        )
        self.past_days = (
            self.declare_parameter("past_days", 2).get_parameter_value().integer_value
        )

    def _handle_location(self, request, response):
        try:
            location = self._get_location()

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name
        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the location: {e}"
            )
            response.ok = False

        return response

    def _handle_current_local_weather(self, request, response):
        self.get_logger().info(f"Fetching current_local_weather")
        try:
            location = self._get_location()
            weather = self._get_weather(location.latitude, location.longitude)
            hour = datetime.now().hour
            if datetime.now().minute > 30:
                hour += 1
            current_weather = weather["hourly"]

            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_celsius = current_weather["temperature_2m"][hour]
            response.feels_like_temperature_celsius = current_weather[
                "apparent_temperature"
            ][hour]
            response.clouds_cover = float(current_weather["cloud_cover"][hour])
            response.wind_speed_kph = current_weather["wind_speed_10m"][hour]
            response.precipitation = current_weather["precipitation"][hour]
            response.snowfall = current_weather["snowfall"][hour]
            response.precipitation_probability_percent = float(
                current_weather["precipitation_probability"][hour]
            )
            response.rain = current_weather["rain"][hour]
            response.is_day = bool(current_weather["is_day"][hour])

        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the local weather: {e}"
            )
            response.ok = False

        self.get_logger().info(f"Returning current_local_weather, ok: {response.ok}")
        return response

    def _handle_local_weather_forecast(self, request, response):
        self.get_logger().info(f"Fetching local_weather_forecast")
        try:
            location = self._get_location()
            weather = self._get_weather(
                location.latitude, location.longitude, daily=True
            )

            daily_data = weather["daily"]
            idx = request.relative_day + self.past_days
            if idx >= len(daily_data["time"]):
                raise ValueError("Invalid relative day")
            response.ok = True
            response.city = location.city
            response.region = location.region
            response.country_code = location.country
            response.country_name = location.country_name

            response.temperature_day_celsius = daily_data["temperature_2m_max"][idx]
            response.temperature_night_celsius = daily_data["temperature_2m_min"][idx]
            response.feals_like_temperature_day_celsius = daily_data[
                "apparent_temperature_max"
            ][idx]
            response.feals_like_temperature_night_celsius = daily_data[
                "apparent_temperature_min"
            ][idx]

            response.precipitation_sum = daily_data["precipitation_sum"][idx]
            response.precipitation_probability_percent = float(
                daily_data["precipitation_probability_mean"][idx]
            )
            response.wind_speed_kph = daily_data["windspeed_10m_max"][idx]
            response.sunrise = datetime.fromisoformat(
                daily_data["sunrise"][idx]
            ).strftime("%H:%M")
            response.sunset = datetime.fromisoformat(
                daily_data["sunset"][idx]
            ).strftime("%H:%M")

        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the local weather forecast: {e}"
            )
            response.ok = False

        self.get_logger().info(f"Returning local_weather_forecast, ok: {response.ok}")
        return response

    def _get_location(self):
        handler = ipinfo.getHandler()
        return handler.getDetails()

    def _get_weather(self, latitude, longitude, daily=False):
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "timezone": "auto",
            "past_days": self.past_days,
        }
        if daily:
            params["daily"] = (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "windspeed_10m_max,"
                "apparent_temperature_max,"
                "apparent_temperature_min,"
                "precipitation_sum,"
                "precipitation_probability_mean,"
                "sunrise,"
                "sunset"
            )

        else:
            params["hourly"] = (
                "temperature_2m,"
                "apparent_temperature,"
                "cloud_cover,"
                "wind_speed_10m,"
                "precipitation,"
                "snowfall,"
                "precipitation_probability,"
                "rain,"
                "is_day"
            )

        response = requests.get(
            OPEN_METEO_API_URL, params=params, timeout=self._timeout
        )
        return json.loads(response.text)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    cloud_data_open_weather_node = OpenWeatherCloudDataNode()
    cloud_data_open_meteo_node = OpenMeteoCloudDataNode()

    executor.add_node(cloud_data_open_weather_node)
    executor.add_node(cloud_data_open_meteo_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cloud_data_open_weather_node.destroy_node()
        cloud_data_open_meteo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
