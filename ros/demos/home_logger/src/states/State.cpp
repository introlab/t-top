#include "State.h"

using namespace std;

StateParameter::StateParameter() {}

StateParameter::~StateParameter() {}

string StateParameter::toString() const
{
    return "";
}


StateType::StateType(type_index type) : m_type(type) {}


State::State(StateManager& stateManager, shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle)
    : m_enabled(false),
      m_stateManager(stateManager),
      m_desireSet(desireSet),
      m_nodeHandle(nodeHandle)
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

void State::onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg) {}

void State::onRobotNameDetected() {}

void State::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg) {}

void State::onAudioAnalysisReceived(const audio_analyzer::AudioAnalysis::ConstPtr& msg) {}

void State::onPersonNamesDetected(const person_identification::PersonNames::ConstPtr& msg) {}

void State::onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg)
{
}

void State::onStateTimeout() {}

void State::onEveryMinuteTimeout() {}

void State::onEveryTenMinutesTimeout() {}


bool containsAtLeastOnePerson(const video_analyzer::VideoAnalysis::ConstPtr& msg)
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
