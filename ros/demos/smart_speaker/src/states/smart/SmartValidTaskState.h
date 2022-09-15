#ifndef SMART_SPEAKER_STATES_SMART_SMART_VALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_VALID_TASK_STATE_H

#include "../common/ValidTaskState.h"

constexpr const char* CURRENT_WEATHER_TASK = "CURRENT_WEATHER";
constexpr const char* DANCE_TASK = "DANCE";

class SmartValidTaskState : public ValidTaskState
{
public:
    SmartValidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SmartValidTaskState() override = default;

    DECLARE_NOT_COPYABLE(SmartValidTaskState);
    DECLARE_NOT_MOVABLE(SmartValidTaskState);

protected:
    std::type_index type() const override;

    void switchState(const std::string& task) override;
};

inline std::type_index SmartValidTaskState::type() const
{
    return std::type_index(typeid(SmartValidTaskState));
}

#endif
