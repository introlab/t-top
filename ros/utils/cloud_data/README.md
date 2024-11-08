# cloud_data

This folder contains a node to get data from the cloud, such as the weather.

## Nodes

### `cloud_data_node.py`

This node gets data location, the current local weather and the local weather forecast from the cloud.

#### Parameters

- `language` (string): The language for the weather description (en or fr). The default value is 'en'.
- `timeout_s` (double): The request timeout in seconds. The default value is 10.0.

#### Services

- `cloud_data/location` ([cloud_data/Location](srv/Location.srv)): The service to get the location from the IP address.
- `cloud_data/current_local_weather` ([cloud_data/CurrentLocalWeather](srv/CurrentLocalWeather.srv)): The service to get
  the current local weather.
- `cloud_data/local_weather_forecast` ([cloud_data/LocalWeatherForecast](srv/LocalWeatherForecast.srv)): The service to
  get the local weather forecast.
