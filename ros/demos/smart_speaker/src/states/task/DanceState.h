#ifndef SMART_SPEAKER_STATES_TASK_DANCE_STATE_H
#define SMART_SPEAKER_STATES_TASK_DANCE_STATE_H

#include "../State.h"

constexpr double DANCE_TIMEOUT_S = 60;

class DanceState : public State
{
    std::type_index m_nextStateType;

    ros::Timer m_timeoutTimer;

public:
    DanceState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~DanceState() override = default;

    DECLARE_NOT_COPYABLE(DanceState);
    DECLARE_NOT_MOVABLE(DanceState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

private:
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index DanceState::type() const
{
    return std::type_index(typeid(DanceState));
}

#endif
