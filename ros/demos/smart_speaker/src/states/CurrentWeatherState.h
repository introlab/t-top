#ifndef SMART_SPEAKER_STATES_CURRENT_WEATHER_STATE_H
#define SMART_SPEAKER_STATES_CURRENT_WEATHER_STATE_H

#include "State.h"

#include <talk/Done.h>

class CurrentWeatherState : public State
{
    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    CurrentWeatherState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~CurrentWeatherState() override = default;

    DECLARE_NOT_COPYABLE(CurrentWeatherState);
    DECLARE_NOT_MOVABLE(CurrentWeatherState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText();
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index CurrentWeatherState::type() const
{
    return std::type_index(typeid(CurrentWeatherState));
}

#endif
