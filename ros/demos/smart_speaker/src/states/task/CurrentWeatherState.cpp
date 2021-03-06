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
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_talkDesireId(MAX_DESIRE_ID)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1, &CurrentWeatherState::talkDoneSubscriberCallback, this);
}

void CurrentWeatherState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto faceFollowingDesire = make_unique<NearestFaceFollowingDesire>();
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
    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::CurrentLocalWeather>("cloud_data/current_local_weather");
    cloud_data::CurrentLocalWeather srv;

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

string CurrentWeatherState::generateEnglishText(bool ok, const cloud_data::CurrentLocalWeather& srv)
{
    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "The current temperature is " << srv.response.temperature_celsius << "??C and ";
        ss << " it feels like " << srv.response.feels_like_temperature_celsius << "??C. ";
        ss << "Humidity is " << srv.response.humidity_percent << "% and ";
        ss << " wind speed is " << srv.response.wind_speed_kph << " kilometers per hour. ";
    }
    else
    {
        ss << "I am not able to get the current weather.";
    }

    return ss.str();
}

string CurrentWeatherState::generateFrenchText(bool ok, const cloud_data::CurrentLocalWeather& srv)
{
    stringstream ss;
    ss.precision(FLOAT_NUMBER_PRECISION);

    if (ok)
    {
        ss << "La temp??rature courante est de " << srv.response.temperature_celsius << "??C. ";
        ss << "La temp??rature courante ressentie est de " << srv.response.feels_like_temperature_celsius << "??C. ";
        ss << "L'humidit?? courante est de " << srv.response.humidity_percent << "%. ";
        ss << "La vitesse courante du vent est de " << srv.response.wind_speed_kph << " kilom??tres par heure. ";
    }
    else
    {
        ss << "Je ne suis pas capable d'obtenir la m??t??o actuelle.";
    }

    return ss.str();
}

void CurrentWeatherState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
