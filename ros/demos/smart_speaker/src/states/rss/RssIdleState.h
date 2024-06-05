#ifndef SMART_SPEAKER_STATES_RSS_RSS_IDLE_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_IDLE_STATE_H

#include "../State.h"

#include <std_msgs/msg/empty.hpp>
#include <perception_msgs/msg/person_names.hpp>

class RssIdleState : public State
{
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_robotNameDetectedSubscriber;
    rclcpp::Subscription<perception_msgs::msg::PersonNames>::SharedPtr m_personNamesSubscriber;

public:
    RssIdleState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node);
    ~RssIdleState() override = default;

    DECLARE_NOT_COPYABLE(RssIdleState);
    DECLARE_NOT_MOVABLE(RssIdleState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;

private:
    void robotNameDetectedSubscriberCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void personNamesSubscriberCallback(const perception_msgs::msg::PersonNames::SharedPtr msg);
};

inline std::type_index RssIdleState::type() const
{
    return std::type_index(typeid(RssIdleState));
}

#endif
