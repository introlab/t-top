#ifndef HOME_LOGGER_STATES_SPECIFIC_WAIT_COMMAND_PARAMETER_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_WAIT_COMMAND_PARAMETER_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/commands/AllCommandParser.h>

class WaitCommandParameterStateParameter : public StateParameter
{
public:
    std::shared_ptr<Command> command;
    std::string parameterName;

    WaitCommandParameterStateParameter();
    WaitCommandParameterStateParameter(std::shared_ptr<Command> command, std::string parameterName);
    ~WaitCommandParameterStateParameter() override;

    std::string toString() const override;
};

class WaitCommandParameterState : public SoundFaceFollowingState
{
    WaitCommandParameterStateParameter m_parameter;

    bool m_transcriptReceived;
    std::string m_parameterResponse;

    std::optional<uint64_t> m_faceAnimationDesireId;
    std::optional<uint64_t> m_speechToTextDesireId;

public:
    WaitCommandParameterState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~WaitCommandParameterState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(WaitCommandParameterState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg) override;
    void onStateTimeout() override;

private:
    void switchState();
};

#endif
