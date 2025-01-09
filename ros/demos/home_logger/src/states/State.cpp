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

void State::onDesireSetChanged([[maybe_unused]] const vector<unique_ptr<Desire>>& desires) {}

void State::onSpeechToTextTranscriptReceived([[maybe_unused]] const perception_msgs::msg::Transcript::SharedPtr& msg) {}

void State::onRobotNameDetected() {}

void State::onVideoAnalysisReceived([[maybe_unused]] const perception_msgs::msg::VideoAnalysis::SharedPtr& msg) {}

void State::onAudioAnalysisReceived([[maybe_unused]] const perception_msgs::msg::AudioAnalysis::SharedPtr& msg) {}

void State::onPersonNamesDetected([[maybe_unused]] const perception_msgs::msg::PersonNames::SharedPtr& msg) {}

void State::onBaseStatusChanged([[maybe_unused]] const daemon_ros_client::msg::BaseStatus::SharedPtr& msg) {}

void State::onStateTimeout() {}

void State::onEveryMinuteTimeout() {}

void State::onEveryTenMinutesTimeout() {}


bool containsAtLeastOnePerson(const perception_msgs::msg::VideoAnalysis::SharedPtr& msg)
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
