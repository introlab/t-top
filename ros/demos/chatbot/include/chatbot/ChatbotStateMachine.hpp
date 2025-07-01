#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <perception_msgs/msg/context_input.hpp>
#include <hbba_lite/core/DesireSet.h>
#include <t_top_hbba_lite/Strategies.h>

class ChatbotStateMachine
{
public:
    ChatbotStateMachine(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<DesireSet> desireSet);

    void start();

private:
    enum class State
    {
        Idle,
        Chat,
        Revive
    };
    State m_state = State::Idle;

    std::shared_ptr<rclcpp::Node> m_node;
    std::shared_ptr<DesireSet> m_desireSet;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_startButtonSubscriber;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_stopButtonSubscriber;
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr m_voiceActivitySubscriber;

    rclcpp::Publisher<perception_msgs::msg::ContextInput>::SharedPtr m_contextInputPublisher;

    rclcpp::TimerBase::SharedPtr m_voiceTimer;

    uint64_t m_chatDesireId = 0;

    void setupSubscribers();
    void enterIdle();
    void enterChat();
    void enterRevive();
    void publishReviveMessage();

    void onStartPressed(const std_msgs::msg::Empty::SharedPtr msg);
    void onStopPressed(const std_msgs::msg::Empty::SharedPtr msg);
    void onVoiceActivity(const std_msgs::msg::UInt8::SharedPtr msg);

    void resetVoiceTimer();
};
