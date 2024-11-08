#ifndef SMART_SPEAKER_STATES_TASK_WEATHER_FORECAST_STATE_H
#define SMART_SPEAKER_STATES_TASK_WEATHER_FORECAST_STATE_H

#include "../common/TalkState.h"

#include <cloud_data/srv/local_weather_forecast.hpp>

class WeatherForecastState : public TalkState
{
    rclcpp::CallbackGroup::SharedPtr m_weatherClientCallbackGroup;
    rclcpp::Client<cloud_data::srv::LocalWeatherForecast>::SharedPtr m_weatherClient;

public:
    WeatherForecastState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node,
        std::type_index nextStateType);
    ~WeatherForecastState() override = default;

    DECLARE_NOT_COPYABLE(WeatherForecastState);
    DECLARE_NOT_MOVABLE(WeatherForecastState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& _) override;
    std::string generateFrenchText(const std::string& _) override;

private:
    void getLocalWeatherForecast(bool& ok, cloud_data::srv::LocalWeatherForecast::Response& response);
};

inline std::type_index WeatherForecastState::type() const
{
    return std::type_index(typeid(WeatherForecastState));
}

#endif
