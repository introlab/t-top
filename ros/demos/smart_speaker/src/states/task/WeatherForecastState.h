#ifndef SMART_SPEAKER_STATES_TASK_WEATHER_FORECAST_STATE_H
#define SMART_SPEAKER_STATES_TASK_WEATHER_FORECAST_STATE_H

#include "../common/TalkState.h"

#include <cloud_data/LocalWeatherForecast.h>
#include <talk/Done.h>

class WeatherForecastState : public TalkState
{
public:
    WeatherForecastState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~WeatherForecastState() override = default;

    DECLARE_NOT_COPYABLE(WeatherForecastState);
    DECLARE_NOT_MOVABLE(WeatherForecastState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& _) override;
    std::string generateFrenchText(const std::string& _) override;

private:
    void getLocalWeatherForecast(bool& ok, cloud_data::LocalWeatherForecast& srv);
};

inline std::type_index WeatherForecastState::type() const
{
    return std::type_index(typeid(WeatherForecastState));
}

#endif
