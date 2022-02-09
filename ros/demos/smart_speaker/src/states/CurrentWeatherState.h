#ifndef SMART_SPEAKER_STATES_CURRENT_WEATHER_STATE_H
#define SMART_SPEAKER_STATES_CURRENT_WEATHER_STATE_H

#include "State.h"

#include <cloud_data/CurrentLocalWeather.h>
#include <talk/Done.h>

class CurrentWeatherState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    CurrentWeatherState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~CurrentWeatherState() override = default;

    DECLARE_NOT_COPYABLE(CurrentWeatherState);
    DECLARE_NOT_MOVABLE(CurrentWeatherState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText();
    std::string generateEnglishText(bool ok, const cloud_data::CurrentLocalWeather& srv);
    std::string generateFrenchText(bool ok, const cloud_data::CurrentLocalWeather& srv);
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index CurrentWeatherState::type() const
{
    return std::type_index(typeid(CurrentWeatherState));
}

#endif
