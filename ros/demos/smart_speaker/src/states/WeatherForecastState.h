#ifndef SMART_SPEAKER_STATES_WEATHER_FORECAST_STATE_H
#define SMART_SPEAKER_STATES_WEATHER_FORECAST_STATE_H

#include "State.h"

#include <talk/Done.h>

class WeatherForecastState : public State
{
    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    WeatherForecastState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~WeatherForecastState() override = default;

    DECLARE_NOT_COPYABLE(WeatherForecastState);
    DECLARE_NOT_MOVABLE(WeatherForecastState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText();
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index WeatherForecastState::type() const
{
    return std::type_index(typeid(WeatherForecastState));
}

#endif
