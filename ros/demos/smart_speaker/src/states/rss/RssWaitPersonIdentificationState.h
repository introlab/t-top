#ifndef SMART_SPEAKER_STATES_RSS_RSS_WAIT_PERSON_IDENTIFICATION_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_WAIT_PERSON_IDENTIFICATION_STATE_H

#include "../State.h"

#include <perception_msgs/msg/person_names.hpp>

class RssWaitPersonIdentificationState : public State
{
    rclcpp::Subscription<perception_msgs::msg::PersonNames>::SharedPtr m_personNamesSubscriber;
    rclcpp::TimerBase::SharedPtr m_timeoutTimer;

public:
    RssWaitPersonIdentificationState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node);
    ~RssWaitPersonIdentificationState() override = default;

    DECLARE_NOT_COPYABLE(RssWaitPersonIdentificationState);
    DECLARE_NOT_MOVABLE(RssWaitPersonIdentificationState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

private:
    void personNamesSubscriberCallback(const perception_msgs::msg::PersonNames::SharedPtr msg);
    void timeoutTimerCallback();
};

inline std::type_index RssWaitPersonIdentificationState::type() const
{
    return std::type_index(typeid(RssWaitPersonIdentificationState));
}

#endif
