#include "CurrentWeatherState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

constexpr chrono::seconds WEATHER_SERVICE_TIMEOUT(20);

CurrentWeatherState::CurrentWeatherState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    type_index nextStateType)
    : TalkState(language, stateManager, desireSet, move(node), nextStateType)
{
    m_weatherClientCallbackGroup = m_node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    m_weatherClient = m_node->create_client<cloud_data::srv::CurrentLocalWeather>(
        "cloud_data/current_local_weather",
        rmw_qos_profile_services_default,
        m_weatherClientCallbackGroup);
}

string CurrentWeatherState::generateEnglishText(const string& _)
{
    bool ok;
    cloud_data::srv::CurrentLocalWeather::Response response;
    getCurrentLocalWeather(ok, response);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "The current temperature is " << response.temperature_celsius << " degree Celsius and ";
        ss << " it feels like " << response.feels_like_temperature_celsius << " degree Celsius. ";
        ss << "Humidity is " << response.humidity_percent << "% and ";
        ss << " wind speed is " << response.wind_speed_kph << " kilometers per hour. ";
    }
    else
    {
        ss << "I am not able to get the current weather.";
    }

    return ss.str();
}

string CurrentWeatherState::generateFrenchText(const string& _)
{
    bool ok;
    cloud_data::srv::CurrentLocalWeather::Response response;
    getCurrentLocalWeather(ok, response);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "La température courante est de " << response.temperature_celsius << " degré Celsius. ";
        ss << "La température courante ressentie est de " << response.feels_like_temperature_celsius
           << " degré Celsius. ";
        ss << "L'humidité courante est de " << response.humidity_percent << "%. ";
        ss << "La vitesse courante du vent est de " << response.wind_speed_kph << " kilomètres par heure. ";
    }
    else
    {
        ss << "Je ne suis pas capable d'obtenir la météo actuelle.";
    }

    return ss.str();
}

void CurrentWeatherState::getCurrentLocalWeather(bool& ok, cloud_data::srv::CurrentLocalWeather::Response& response)
{
    auto request = make_shared<cloud_data::srv::CurrentLocalWeather::Request>();
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
