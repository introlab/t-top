#ifndef SMART_SPEAKER_STATES_TASK_CURRENT_WEATHER_STATE_H
#define SMART_SPEAKER_STATES_TASK_CURRENT_WEATHER_STATE_H

#include "../common/TalkState.h"

#include <cloud_data/CurrentLocalWeather.h>

class CurrentWeatherState : public TalkState
{
public:
    CurrentWeatherState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~CurrentWeatherState() override = default;

    DECLARE_NOT_COPYABLE(CurrentWeatherState);
    DECLARE_NOT_MOVABLE(CurrentWeatherState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& _) override;
    std::string generateFrenchText(const std::string& _) override;

private:
    void getCurrentLocalWeather(bool& ok, cloud_data::CurrentLocalWeather& srv);
};

inline std::type_index CurrentWeatherState::type() const
{
    return std::type_index(typeid(CurrentWeatherState));
}

#endif
