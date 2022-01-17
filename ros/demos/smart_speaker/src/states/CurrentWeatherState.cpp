#include "CurrentWeatherState.h"
#include "StateManager.h"
#include "IdleState.h"

#include <cloud_data/CurrentLocalWeather.h>

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

CurrentWeatherState::CurrentWeatherState(StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(stateManager, desireSet, nodeHandle),
        m_talkDesireId(MAX_DESIRE_ID)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1,
        &CurrentWeatherState::talkDoneSubscriberCallback, this);
}

void CurrentWeatherState::enable(const string& parameter)
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

void CurrentWeatherState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
}

string CurrentWeatherState::generateText()
{
    stringstream ss;
    ss.precision(3);

    ros::ServiceClient service = m_nodeHandle.serviceClient<cloud_data::CurrentLocalWeather>("cloud_data/current_local_weather");
    cloud_data::CurrentLocalWeather srv;
    if (service.exists() && service.call(srv))
    {
        ss << "The current temperature is " << srv.response.temperature_celsius << "°C. ";
        ss << "The current feels like temperature is " << srv.response.feels_like_temperature_celsius << "°C. ";
        ss << "The current humidity is " << srv.response.humidity_percent << "%. ";
        ss << "The current wind speed is " << srv.response.wind_speed_kph << " kilometers per hour. ";

        if (srv.response.wind_gust_kph != -1)
        {
            ss << "The current wind gust speed is " << srv.response.wind_gust_kph << " kilometers per hour. ";
        }
    }
    else
    {
        ss << "I am not able to get the current weather.";
    }

    return ss.str();
}

void CurrentWeatherState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_stateManager.switchTo<IdleState>();
}
