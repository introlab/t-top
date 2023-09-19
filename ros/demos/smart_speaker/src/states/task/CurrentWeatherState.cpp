#include "CurrentWeatherState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

CurrentWeatherState::CurrentWeatherState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType)
    : TalkState(language, stateManager, desireSet, nodeHandle, nextStateType)
{
}

string CurrentWeatherState::generateEnglishText(const string& _)
{
    bool ok;
    cloud_data::CurrentLocalWeather srv;
    getCurrentLocalWeather(ok, srv);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "The current temperature is " << srv.response.temperature_celsius << " degree Celsius and ";
        ss << " it feels like " << srv.response.feels_like_temperature_celsius << " degree Celsius. ";
        ss << "Humidity is " << srv.response.humidity_percent << "% and ";
        ss << " wind speed is " << srv.response.wind_speed_kph << " kilometers per hour. ";
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
    cloud_data::CurrentLocalWeather srv;
    getCurrentLocalWeather(ok, srv);

    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "La température courante est de " << srv.response.temperature_celsius << " degré Celsius. ";
        ss << "La température courante ressentie est de " << srv.response.feels_like_temperature_celsius << " degré Celsius. ";
        ss << "L'humidité courante est de " << srv.response.humidity_percent << "%. ";
        ss << "La vitesse courante du vent est de " << srv.response.wind_speed_kph << " kilomètres par heure. ";
    }
    else
    {
        ss << "Je ne suis pas capable d'obtenir la météo actuelle.";
    }

    return ss.str();
}

void CurrentWeatherState::getCurrentLocalWeather(bool& ok, cloud_data::CurrentLocalWeather& srv)
{
    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::CurrentLocalWeather>("cloud_data/current_local_weather");

    ok = service.exists() && service.call(srv);
}
