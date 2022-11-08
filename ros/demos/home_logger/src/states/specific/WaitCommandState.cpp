#include "WaitCommandState.h"
#include "IdleState.h"
#include "ExecuteCommandState.h"
#include "../StateManager.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringResources.h>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

WaitCommandState::WaitCommandState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_transcriptReceived(false)
{
}

WaitCommandState::~WaitCommandState() {}

void WaitCommandState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_transcriptReceived = false;
    m_commands.clear();

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("blink");
    m_speechToTextDesireId = m_desireSet->addDesire<SpeechToTextDesire>();
}

void WaitCommandState::onDisabling()
{
    SoundFaceFollowingState::onDisabling();

    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = nullopt;
    }
    if (m_speechToTextDesireId.has_value())
    {
        m_desireSet->removeDesire(m_speechToTextDesireId.value());
        m_speechToTextDesireId = nullopt;
    }
}

void WaitCommandState::onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg)
{
    m_transcriptReceived = true;
    m_commands = m_parser.parse(msg->text);

    if (msg->is_final)
    {
        switchState();
    }
}

void WaitCommandState::onStateTimeout()
{
    switchState();
}

void WaitCommandState::switchState()
{
    if (!m_transcriptReceived)
    {
        m_stateManager.switchTo<IdleState>();
    }
    else if (m_commands.empty())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.wait_command_state.invalid_command"),
            "no",
            "sad",
            StateType::get<WaitCommandState>()));
    }
    else if (m_commands.size() > 1)
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.wait_command_state.many_commands"),
            "",  // No gesture
            "skeptic",
            StateType::get<WaitCommandState>()));
    }
    else if (m_commands[0]->type() == CommandType::get<NothingCommand>())
    {
        m_stateManager.switchTo<IdleState>();
    }
    else
    {
        m_stateManager.switchTo<ExecuteCommandState>(ExecuteCommandStateParameter(move(m_commands[0])));
    }
}
