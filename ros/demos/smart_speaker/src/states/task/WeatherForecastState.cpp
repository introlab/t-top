#include "WeatherForecastState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

constexpr chrono::seconds WEATHER_SERVICE_TIMEOUT(20);

WeatherForecastState::WeatherForecastState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    std::type_index nextStateType)
    : TalkState(language, stateManager, desireSet, move(node), nextStateType)
{
    m_weatherClientCallbackGroup = m_node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    m_weatherClient = m_node->create_client<cloud_data::srv::LocalWeatherForecast>(
        "cloud_data/local_weather_forecast",
        rmw_qos_profile_services_default,
        m_weatherClientCallbackGroup);
}

string WeatherForecastState::generateEnglishText(const string& _)
{
    bool ok;
    cloud_data::srv::LocalWeatherForecast::Response response;
    getLocalWeatherForecast(ok, response);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "Tomorrow morning, the temperature will be " << response.temperature_morning_celsius
           << " degree Celsius and ";
        ss << "the feels like temperature will be " << response.feals_like_temperature_morning_celsius
           << " degree Celsius. ";

        ss << "During the day of tomorrow, the temperature will be " << response.temperature_day_celsius
           << " degree Celsius and ";
        ss << "the feels like temperature will be " << response.feals_like_temperature_day_celsius
           << " degree Celsius. ";

        ss << "Tomorrow evening, the temperature will be " << response.temperature_evening_celsius
           << " degree Celsius and ";
        ss << "the feels like temperature will be " << response.feals_like_temperature_evening_celsius
           << " degree Celsius. ";

        ss << "Tomorrow night, the temperature will be " << response.temperature_night_celsius
           << " degree Celsius and ";
        ss << "the feels like temperature will be " << response.feals_like_temperature_night_celsius
           << " degree Celsius. ";

        ss << "Tomorrow, the humidity will be " << response.humidity_percent << "%, ";
        ss << "the wind speed will be " << response.wind_speed_kph << " kilometers per hour,";
    }
    else
    {
        ss << "I am not able to get the weather forecast.";
    }

    return ss.str();
}

string WeatherForecastState::generateFrenchText(const string& _)
{
    bool ok;
    cloud_data::srv::LocalWeatherForecast::Response response;
    getLocalWeatherForecast(ok, response);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "Demain matin, la température sera de " << response.temperature_morning_celsius << " degré Celsius et ";
        ss << "la température ressentie sera de " << response.feals_like_temperature_morning_celsius
           << " degré Celsius. ";

        ss << "Demain en journée, la température sera de " << response.temperature_day_celsius << " degré Celsius et ";
        ss << "la température ressentie sera de " << response.feals_like_temperature_day_celsius << " degré Celsius. ";

        ss << "Demain en soirée, la température sera de " << response.temperature_evening_celsius
           << " degré Celsius et ";
        ss << "la température ressentie sera de " << response.feals_like_temperature_evening_celsius
           << " degré Celsius. ";

        ss << "La nuit prochaine, la température sera de " << response.temperature_night_celsius
           << " degré Celsius et ";
        ss << "la température ressentie sera de " << response.feals_like_temperature_night_celsius
           << " degré Celsius. ";

        ss << "Demain, l'humidité sera de " << response.humidity_percent << "%, ";
        ss << "la vitesse du vent sera de " << response.wind_speed_kph << " kilomètres par heure, ";
    }
    else
    {
        ss << "Je ne suis pas capable d'obtenir les prévisions météo.";
    }

    return ss.str();
}

void WeatherForecastState::getLocalWeatherForecast(bool& ok, cloud_data::srv::LocalWeatherForecast::Response& response)
{
    auto request = make_shared<cloud_data::srv::LocalWeatherForecast::Request>();
    request->relative_day = 1;  // tomorow

    auto future = m_weatherClient->async_send_request(request);
    auto status = future.wait_for(WEATHER_SERVICE_TIMEOUT);
    if (status == future_status::ready)
    {
        response = *future.get();
        ok = response.ok;
    }
    else
    {
        ok = false;
    }
}
