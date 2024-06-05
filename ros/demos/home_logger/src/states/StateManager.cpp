#include "StateManager.h"

using namespace std;

const int TIMEOUT_S = 30;
const int ONE_MINUTE_S = 60;
const int TEN_MINUTES_S = 600;

StateManager::StateManager(shared_ptr<DesireSet> desireSet, rclcpp::Node::SharedPtr node)
    : m_desireSet(desireSet),
      m_node(move(node)),
      m_currentState(nullptr)
{
    m_desireSet->addObserver(this);

    m_speechToTextSubscriber = m_node->create_subscription<perception_msgs::msg::Transcript>(
        "speech_to_text/transcript",
        1,
        [this](const perception_msgs::msg::Transcript::SharedPtr msg) { onSpeechToTextTranscriptReceived(msg); });
    m_robotNameDetectedSubscriber = m_node->create_subscription<std_msgs::msg::Empty>(
        "robot_name_detected",
        1,
        [this](const std_msgs::msg::Empty::SharedPtr msg) { onRobotNameDetected(msg); });
    m_videoAnalysisSubscriber = m_node->create_subscription<perception_msgs::msg::VideoAnalysis>(
        "video_analysis",
        1,
        [this](const perception_msgs::msg::VideoAnalysis::SharedPtr msg) { onVideoAnalysisReceived(msg); });
    m_audioAnalysisSubscriber = m_node->create_subscription<perception_msgs::msg::AudioAnalysis>(
        "audio_analysis",
        1,
        [this](const perception_msgs::msg::AudioAnalysis::SharedPtr msg) { onAudioAnalysisReceived(msg); });
    m_personNamesSubscriber = m_node->create_subscription<perception_msgs::msg::PersonNames>(
        "person_names",
        1,
        [this](const perception_msgs::msg::PersonNames::SharedPtr msg) { onPersonNamesDetected(msg); });
    m_baseStatusSubscriber = m_node->create_subscription<daemon_ros_client::msg::BaseStatus>(
        "daemon/base_status",
        1,
        [this](const daemon_ros_client::msg::BaseStatus::SharedPtr msg) { onBaseStatusChanged(msg); });

    m_everyMinuteTimer = m_node->create_wall_timer(chrono::seconds(ONE_MINUTE_S), [this]() { onEveryMinuteTimeout(); });
    m_everyTenMinuteTimer =
        m_node->create_wall_timer(chrono::seconds(TEN_MINUTES_S), [this]() { onEveryTenMinutesTimeout(); });
}

StateManager::~StateManager()
{
    m_desireSet->removeObserver(this);
}

void StateManager::addState(unique_ptr<State> state)
{
    m_states.emplace(state->type(), move(state));
}

void StateManager::switchTo(StateType type, const StateParameter& parameter)
{
    if (m_stateTimeoutTimer)
    {
        m_stateTimeoutTimer->cancel();
        m_stateTimeoutTimer = nullptr;
    }

    auto transaction = m_desireSet->beginTransaction();

    StateType previousStateType = StateType::null();
    if (m_currentState != nullptr)
    {
        RCLCPP_INFO_STREAM(m_node->get_logger(), "Disabling " << m_currentState->type().name());
        m_currentState->disable();
        previousStateType = m_currentState->type();
    }

    RCLCPP_INFO_STREAM(m_node->get_logger(), "Enabling " << type.name() << "(" << parameter.toString() << ")");
    m_currentState = m_states.at(type).get();
    m_currentState->enable(parameter, previousStateType);

    m_stateTimeoutTimer = m_node->create_wall_timer(chrono::seconds(TIMEOUT_S), [this]() { onStateTimeout(); });
}

void StateManager::onDesireSetChanged(const vector<unique_ptr<Desire>>& desires)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onDesireSetChanged(desires);
    }
}

void StateManager::onSpeechToTextTranscriptReceived(const perception_msgs::msg::Transcript::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onSpeechToTextTranscriptReceived(msg);
    }
}

void StateManager::onRobotNameDetected(const std_msgs::msg::Empty::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onRobotNameDetected();
    }
}

void StateManager::onVideoAnalysisReceived(const perception_msgs::msg::VideoAnalysis::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onVideoAnalysisReceived(msg);
    }
}

void StateManager::onAudioAnalysisReceived(const perception_msgs::msg::AudioAnalysis::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onAudioAnalysisReceived(msg);
    }
}

void StateManager::onPersonNamesDetected(const perception_msgs::msg::PersonNames::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onPersonNamesDetected(msg);
    }
}

void StateManager::onBaseStatusChanged(const daemon_ros_client::msg::BaseStatus::SharedPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onBaseStatusChanged(msg);
    }
}

void StateManager::onStateTimeout()
{
    if (m_currentState != nullptr)
    {
        m_currentState->onStateTimeout();
    }
}

void StateManager::onEveryMinuteTimeout()
{
    if (m_currentState != nullptr)
    {
        m_currentState->onEveryMinuteTimeout();
    }
}

void StateManager::onEveryTenMinutesTimeout()
{
    if (m_currentState != nullptr)
    {
        m_currentState->onEveryTenMinutesTimeout();
    }
}
