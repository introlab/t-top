#include "WaitCommandParameterState.h"
#include "ExecuteCommandState.h"
#include "IdleState.h"

#include <home_logger_common/language/StringResources.h>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

WaitCommandParameterStateParameter::WaitCommandParameterStateParameter() {}

WaitCommandParameterStateParameter::WaitCommandParameterStateParameter(
    shared_ptr<Command> command,
    string parameterName)
    : command(move(command)),
      parameterName(move(parameterName))
{
}

WaitCommandParameterStateParameter::~WaitCommandParameterStateParameter() {}

string WaitCommandParameterStateParameter::toString() const
{
    stringstream ss;
    ss << "command=" << command->type().name();
    ss << ", parameter_name=" << parameterName;
    return ss.str();
}


WaitCommandParameterState::WaitCommandParameterState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_transcriptReceived(false)
{
}

WaitCommandParameterState::~WaitCommandParameterState() {}

void WaitCommandParameterState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_parameter = dynamic_cast<const WaitCommandParameterStateParameter&>(parameter);

    m_transcriptReceived = false;
    m_parameterResponse = "";

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("blink");
    m_speechToTextDesireId = m_desireSet->addDesire<SpeechToTextDesire>();
}

void WaitCommandParameterState::onDisabling()
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

void WaitCommandParameterState::onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg)
{
    m_transcriptReceived = true;
    m_parameterResponse = msg->text;

    if (msg->is_final)
    {
        switchState();
    }
}

void WaitCommandParameterState::onStateTimeout()
{
    switchState();
}

void WaitCommandParameterState::switchState()
{
    if (!m_transcriptReceived)
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.wait_command_parameter_state.timeout"),
            "no",
            "sad",
            StateType::get<IdleState>()));
    }
    else
    {
        m_stateManager.switchTo<ExecuteCommandState>(
            ExecuteCommandStateParameter(m_parameter.command, m_parameter.parameterName, m_parameterResponse));
    }
}
