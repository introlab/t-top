#include <chatbot/ChatbotStateMachine.hpp>
#include <perception_msgs/msg/context_input.hpp>
#include <t_top_hbba_lite/Desires.h>

using namespace std::chrono_literals;

ChatbotStateMachine::ChatbotStateMachine(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<DesireSet> desireSet)
    : m_node(std::move(node)),
      m_desireSet(std::move(desireSet))
{
}

void ChatbotStateMachine::start()
{
    setupSubscribers();
    enterIdle();
}

void ChatbotStateMachine::setupSubscribers()
{
    m_startButtonSubscriber = m_node->create_subscription<std_msgs::msg::Empty>(
        "/daemon/start_button_pressed",
        1,
        std::bind(&ChatbotStateMachine::onStartPressed, this, std::placeholders::_1));
    m_stopButtonSubscriber = m_node->create_subscription<std_msgs::msg::Empty>(
        "/daemon/stop_button_pressed",
        1,
        std::bind(&ChatbotStateMachine::onStopPressed, this, std::placeholders::_1));

    m_voiceActivitySubscriber = m_node->create_subscription<std_msgs::msg::UInt8>(
        "voice_activity",
        10,
        std::bind(&ChatbotStateMachine::onVoiceActivity, this, std::placeholders::_1));

    m_contextInputPublisher = m_node->create_publisher<perception_msgs::msg::ContextInput>(
        "chat/context_input",
        rclcpp::QoS(1).transient_local());

    m_voiceTimer = m_node->create_wall_timer(20s, std::bind(&ChatbotStateMachine::enterRevive, this));
    m_voiceTimer->cancel();
}

void ChatbotStateMachine::onStartPressed(const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;
    RCLCPP_INFO(m_node->get_logger(), "Start button pressed");
    enterChat();
}

void ChatbotStateMachine::onStopPressed(const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;
    RCLCPP_INFO(m_node->get_logger(), "Stop button pressed");
    enterIdle();
}

void ChatbotStateMachine::onVoiceActivity(const std_msgs::msg::UInt8::SharedPtr msg)
{
    (void)msg;
    resetVoiceTimer();
}

void ChatbotStateMachine::resetVoiceTimer()
{
    if (m_state == State::Chat)
    {
        m_voiceTimer->reset();
    }
}

void ChatbotStateMachine::enterIdle()
{
    RCLCPP_INFO(m_node->get_logger(), "Entering Idle state");
    m_state = State::Idle;

    if (m_chatDesireId != 0)
    {
        m_desireSet->removeDesire(m_chatDesireId);
        m_chatDesireId = 0;
    }

    m_voiceTimer->cancel();
}

void ChatbotStateMachine::enterChat()
{
    if (m_state == State::Chat)
        return;

    RCLCPP_INFO(m_node->get_logger(), "Entering Chat state");
    m_state = State::Chat;

    if (m_chatDesireId == 0)
    {
        auto chatDesire = std::make_unique<ChatDesire>();
        m_chatDesireId = m_desireSet->addDesire(std::move(chatDesire));
    }

    m_voiceTimer->reset();
}

void ChatbotStateMachine::enterRevive()
{
    if (m_state != State::Chat)
        return;
    RCLCPP_INFO(m_node->get_logger(), "No voice activity detected. Entering Revive state.");
    m_state = State::Revive;

    if (m_chatDesireId != 0)
    {
        m_desireSet->removeDesire(m_chatDesireId);
        m_chatDesireId = 0;
    }

    publishReviveMessage();
    enterChat();
}

void ChatbotStateMachine::publishReviveMessage()
{
    perception_msgs::msg::ContextInput msg;
    // msg.context_type = "system";
    // msg.value = "revive";
    // msg.origin = "FSM";  // Replace with the correct field name if different
    m_contextInputPublisher->publish(msg);
}
