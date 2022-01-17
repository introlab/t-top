#include "WeatherForecastState.h"
#include "StateManager.h"
#include "IdleState.h"

#include <cloud_data/LocalWeatherForecast.h>

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

WeatherForecastState::WeatherForecastState(StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(stateManager, desireSet, nodeHandle),
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
    stringstream ss;
    ss.precision(3);

    ros::ServiceClient service = m_nodeHandle.serviceClient<cloud_data::LocalWeatherForecast>("cloud_data/local_weather_forecast");
    cloud_data::LocalWeatherForecast srv;
    srv.request.relative_day = 1; // tomorow
    if (service.exists() && service.call(srv))
    {
        ss << "Tomorrow morning, the temperature will be " << srv.response.temperature_morning_celsius << " °C. ";
        ss << "The feels like temperature will be " << srv.response.feals_like_temperature_morning_celsius << " °C. ";

        ss << "During the day of tomorrow, the temperature will be " << srv.response.temperature_day_celsius << " °C. ";
        ss << "The feels like temperature will be " << srv.response.feals_like_temperature_day_celsius << " °C. ";

        ss << "Tomorrow evening, the temperature will be " << srv.response.temperature_evening_celsius << " °C. ";
        ss << "The feels like temperature will be " << srv.response.feals_like_temperature_evening_celsius << " °C. ";

        ss << "Tomorrow night, the temperature will be " << srv.response.temperature_night_celsius << " °C. ";
        ss << "The feels like temperature will be " << srv.response.feals_like_temperature_night_celsius << " °C. ";

        ss << "Tomorrow, the humidity will be " << srv.response.humidity_percent << "%, ";
        ss << "the wind speed will be " << srv.response.wind_speed_kph << " kilometers per hour, ";

        if (srv.response.wind_gust_kph != -1)
        {
            ss << "the wind gust speed will be " << srv.response.wind_gust_kph << " kilometers per hour. ";
        }
    }
    else
    {
        ss << "I am not able to get the weather forecast.";
    }

    return ss.str();
}

void WeatherForecastState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_stateManager.switchTo<IdleState>();
}
