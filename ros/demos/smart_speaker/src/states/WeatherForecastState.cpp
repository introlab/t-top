#include "WeatherForecastState.h"
#include "StateManager.h"

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

WeatherForecastState::WeatherForecastState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    std::type_index nextStateType) :
        State(language, stateManager, desireSet, nodeHandle),
        m_nextStateType(nextStateType),
        m_talkDesireId(MAX_DESIRE_ID)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1,
        &WeatherForecastState::talkDoneSubscriberCallback, this);
}

void WeatherForecastState::enable(const string& parameter)
{
    State::enable(parameter);

    auto faceFollowingDesire = make_unique<FaceFollowingDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");
    auto talkDesire = make_unique<TalkDesire>(generateText());
    m_talkDesireId = talkDesire->id();

    m_desireIds.emplace_back(faceFollowingDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());
    m_desireIds.emplace_back(talkDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(faceFollowingDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
    m_desireSet->addDesire(move(talkDesire));
}

void WeatherForecastState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
}

string WeatherForecastState::generateText()
{
    ros::ServiceClient service = m_nodeHandle.serviceClient<cloud_data::LocalWeatherForecast>("cloud_data/local_weather_forecast");
    cloud_data::LocalWeatherForecast srv;
    srv.request.relative_day = 1; // tomorow
    bool ok = service.exists() && service.call(srv);

    switch (language())
    {
    case Language::ENGLISH:
        return generateEnglishText(ok, srv);
    case Language::FRENCH:
        return generateFrenchText(ok, srv);
    }

    return "";
}

string WeatherForecastState::generateEnglishText(bool ok, const cloud_data::LocalWeatherForecast& srv)
{
    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "Tomorrow morning, the temperature will be " << srv.response.temperature_morning_celsius << " °C and ";
        ss << "the feels like temperature will be " << srv.response.feals_like_temperature_morning_celsius << " °C. ";

        ss << "During the day of tomorrow, the temperature will be " << srv.response.temperature_day_celsius << " °C and ";
        ss << "the feels like temperature will be " << srv.response.feals_like_temperature_day_celsius << " °C. ";

        ss << "Tomorrow evening, the temperature will be " << srv.response.temperature_evening_celsius << " °C and ";
        ss << "the feels like temperature will be " << srv.response.feals_like_temperature_evening_celsius << " °C. ";

        ss << "Tomorrow night, the temperature will be " << srv.response.temperature_night_celsius << " °C and ";
        ss << "the feels like temperature will be " << srv.response.feals_like_temperature_night_celsius << " °C. ";

        ss << "Tomorrow, the humidity will be " << srv.response.humidity_percent << "%, ";
        ss << "the wind speed will be " << srv.response.wind_speed_kph << " kilometers per hour,";
    }
    else
    {
        ss << "I am not able to get the weather forecast.";
    }

    return ss.str();
}

string WeatherForecastState::generateFrenchText(bool ok, const cloud_data::LocalWeatherForecast& srv)
{
    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "Demain matin, la température sera de " << srv.response.temperature_morning_celsius << " °C et ";
        ss << "la température ressentie sera de " << srv.response.feals_like_temperature_morning_celsius << " °C. ";

        ss << "Demain en journée, la température sera de " << srv.response.temperature_day_celsius << " °C et ";
        ss << "la température ressentie sera de " << srv.response.feals_like_temperature_day_celsius << " °C. ";

        ss << "Demain en soirée, la température sera de " << srv.response.temperature_evening_celsius << " °C et ";
        ss << "la température ressentie sera de " << srv.response.feals_like_temperature_evening_celsius << " °C. ";

        ss << "La nuit prochaine, la température sera de " << srv.response.temperature_night_celsius << " °C et ";
        ss << "la température ressentie sera de " << srv.response.feals_like_temperature_night_celsius << " °C. ";

        ss << "Demain, l'humidité sera de " << srv.response.humidity_percent << "%, ";
        ss << "la vitesse du vent sera de " << srv.response.wind_speed_kph << " kilomètres par heure, ";
    }
    else
    {
        ss << "Je ne suis pas capable d'obtenir les prévisions météo.";
    }

    return ss.str();
}

void WeatherForecastState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
