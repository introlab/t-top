#ifndef SMART_SPEAKER_STATES_WEATHER_FORECAST_STATE_H
#define SMART_SPEAKER_STATES_WEATHER_FORECAST_STATE_H

#include "State.h"

#include <cloud_data/LocalWeatherForecast.h>
#include <talk/Done.h>

class WeatherForecastState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    WeatherForecastState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~WeatherForecastState() override = default;

    DECLARE_NOT_COPYABLE(WeatherForecastState);
    DECLARE_NOT_MOVABLE(WeatherForecastState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText();
    std::string generateEnglishText(bool ok, const cloud_data::LocalWeatherForecast& srv);
    std::string generateFrenchText(bool ok, const cloud_data::LocalWeatherForecast& srv);
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index WeatherForecastState::type() const
{
    return std::type_index(typeid(WeatherForecastState));
}

#endif
