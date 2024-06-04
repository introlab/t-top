#include "State.h"

using namespace std;

StateParameter::StateParameter() {}

StateParameter::~StateParameter() {}

string StateParameter::toString() const
{
    return "";
}


StateType::StateType(type_index type) : m_type(type) {}


State::State(StateManager& stateManager, shared_ptr<DesireSet> desireSet, rclcpp::Node::SharedPtr node)
    : m_enabled(false),
      m_stateManager(stateManager),
      m_desireSet(desireSet),
      m_node(move(node))
{
}

State::~State() {}

void State::enable(const StateParameter& parameter, const StateType& previousStageType)
{
    if (!m_enabled)
    {
        m_enabled = true;
        onEnabling(parameter, previousStageType);
    }
}

void State::disable()
{
    if (m_enabled)
    {
        m_enabled = false;
        onDisabling();
    }
}

void State::onDesireSetChanged(const vector<unique_ptr<Desire>>& desires) {}

void State::onSpeechToTextTranscriptReceived(const speech_to_text::msg::Transcript::SharedPtr& msg) {}

void State::onRobotNameDetected() {}

void State::onVideoAnalysisReceived(const video_analyzer::msg::VideoAnalysis::SharedPtr& msg) {}

void State::onAudioAnalysisReceived(const audio_analyzer::msg::AudioAnalysis::SharedPtr& msg) {}

void State::onPersonNamesDetected(const person_identification::msg::PersonNames::SharedPtr& msg) {}

void State::onBaseStatusChanged(const daemon_ros_client::msg::BaseStatus::SharedPtr& msg) {}

void State::onStateTimeout() {}

void State::onEveryMinuteTimeout() {}

void State::onEveryTenMinutesTimeout() {}


bool containsAtLeastOnePerson(const video_analyzer::msg::VideoAnalysis::SharedPtr& msg)
{
    for (auto& object : msg->objects)
    {
        if (object.object_class == "person")
        {
            return true;
        }
    }

    return false;
}
