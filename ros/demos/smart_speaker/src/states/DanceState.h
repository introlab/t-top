#ifndef SMART_SPEAKER_STATES_DANCE_STATE_H
#define SMART_SPEAKER_STATES_DANCE_STATE_H

#include "State.h"

constexpr double DANCE_TIMEOUT_S = 60;

class DanceState : public State
{
    ros::Timer m_timeoutTimer;

public:
    DanceState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~DanceState() override = default;

    DECLARE_NOT_COPYABLE(DanceState);
    DECLARE_NOT_MOVABLE(DanceState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index DanceState::type() const
{
    return std::type_index(typeid(DanceState));
}

#endif
