#include "StateManager.h"

using namespace std;

const double TIMEOUT_S = 30.0;
const double ONE_MINUTE_S = 60.0;
const double TEN_MINUTES_S = 600.0;

StateManager::StateManager(shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle)
    : m_desireSet(desireSet),
      m_nodeHandle(nodeHandle),
      m_currentState(nullptr)
{
    m_desireSet->addObserver(this);

    m_speechToTextSubscriber =
        nodeHandle.subscribe("speech_to_text/transcript", 1, &StateManager::onSpeechToTextTranscriptReceived, this);
    m_robotNameDetectedSubscriber =
        nodeHandle.subscribe("robot_name_detected", 1, &StateManager::onRobotNameDetected, this);
    m_videoAnalysisSubscriber = nodeHandle.subscribe("video_analysis", 1, &StateManager::onVideoAnalysisReceived, this);
    m_audioAnalysisSubscriber = nodeHandle.subscribe("audio_analysis", 1, &StateManager::onAudioAnalysisReceived, this);
    m_personNamesSubscriber = nodeHandle.subscribe("person_names", 1, &StateManager::onPersonNamesDetected, this);
    m_baseStatusSubscriber = nodeHandle.subscribe("daemon/base_status", 1, &StateManager::onBaseStatusChanged, this);

    m_everyMinuteTimer =
        m_nodeHandle.createTimer(ros::Duration(ONE_MINUTE_S), &StateManager::onEveryMinuteTimeout, this);
    m_everyTenMinuteTimer =
        m_nodeHandle.createTimer(ros::Duration(TEN_MINUTES_S), &StateManager::onEveryTenMinutesTimeout, this);
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
    if (m_stateTimeoutTimer.isValid())
    {
        m_stateTimeoutTimer.stop();
    }

    auto transaction = m_desireSet->beginTransaction();

    StateType previousStateType = StateType::null();
    if (m_currentState != nullptr)
    {
        ROS_INFO_STREAM("Disabling " << m_currentState->type().name());
        m_currentState->disable();
        previousStateType = m_currentState->type();
    }

    ROS_INFO_STREAM("Enabling " << type.name() << "(" << parameter.toString() << ")");
    m_currentState = m_states.at(type).get();
    m_currentState->enable(parameter, previousStateType);

    constexpr bool ONE_SHOT = true;
    m_stateTimeoutTimer =
        m_nodeHandle.createTimer(ros::Duration(TIMEOUT_S), &StateManager::onStateTimeout, this, ONE_SHOT);
}

void StateManager::onDesireSetChanged(const vector<unique_ptr<Desire>>& desires)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onDesireSetChanged(desires);
    }
}

void StateManager::onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onSpeechToTextTranscriptReceived(msg);
    }
}

void StateManager::onRobotNameDetected(const std_msgs::Empty::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onRobotNameDetected();
    }
}

void StateManager::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onVideoAnalysisReceived(msg);
    }
}

void StateManager::onAudioAnalysisReceived(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onAudioAnalysisReceived(msg);
    }
}

void StateManager::onPersonNamesDetected(const person_identification::PersonNames::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onPersonNamesDetected(msg);
    }
}

void StateManager::onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onBaseStatusChanged(msg);
    }
}

void StateManager::onStateTimeout(const ros::TimerEvent& event)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onStateTimeout();
    }
}

void StateManager::onEveryMinuteTimeout(const ros::TimerEvent& event)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onEveryMinuteTimeout();
    }
}

void StateManager::onEveryTenMinutesTimeout(const ros::TimerEvent& event)
{
    if (m_currentState != nullptr)
    {
        m_currentState->onEveryTenMinutesTimeout();
    }
}
