#ifndef SMART_SPEAKER_STATES_COMMON_AFTER_TASK_DELAY_STATE_H
#define SMART_SPEAKER_STATES_COMMON_AFTER_TASK_DELAY_STATE_H

#include "../State.h"

#include <std_msgs/msg/empty.hpp>

class AfterTaskDelayState : public State
{
    std::type_index m_nextStateType;

    bool m_useAfterTaskDelayDurationTopic;
    std::chrono::milliseconds m_durationMs;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_startButtonSubscriber;
    rclcpp::TimerBase::SharedPtr m_timeoutTimer;

public:
    AfterTaskDelayState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node,
        std::type_index nextStateType,
        bool useAfterTaskDelayDurationTopic,
        std::chrono::milliseconds durationMs);
    ~AfterTaskDelayState() override = default;

    DECLARE_NOT_COPYABLE(AfterTaskDelayState);
    DECLARE_NOT_MOVABLE(AfterTaskDelayState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

private:
    void startButtonCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void timeoutTimerCallback();
};

inline std::type_index AfterTaskDelayState::type() const
{
    return std::type_index(typeid(AfterTaskDelayState));
}

#endif
