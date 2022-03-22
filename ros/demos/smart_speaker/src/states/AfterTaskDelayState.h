#ifndef SMART_SPEAKER_AFTER_TASK_DELAY_STATE_H
#define SMART_SPEAKER_AFTER_TASK_DELAY_STATE_H

#include "State.h"

class AfterTaskDelayState : public State
{
    std::type_index m_nextStateType;

    ros::Duration m_duration;
    ros::Timer m_timeoutTimer;

public:
    AfterTaskDelayState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType,
        ros::Duration duration);
    ~AfterTaskDelayState() override = default;

    DECLARE_NOT_COPYABLE(AfterTaskDelayState);
    DECLARE_NOT_MOVABLE(AfterTaskDelayState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index AfterTaskDelayState::type() const
{
    return std::type_index(typeid(AfterTaskDelayState));
}

#endif
