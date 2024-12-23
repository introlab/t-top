#ifndef HOME_LOGGER_STATES_SPECIFIC_WAIT_COMMAND_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_WAIT_COMMAND_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/commands/AllCommandParser.h>

class WaitCommandState : public SoundFaceFollowingState
{
    AllCommandParser m_parser;
    bool m_transcriptReceived;
    std::vector<std::unique_ptr<Command>> m_commands;

    std::optional<uint64_t> m_faceAnimationDesireId;
    std::optional<uint64_t> m_speechToTextDesireId;

public:
    WaitCommandState(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, rclcpp::Node::SharedPtr node);
    ~WaitCommandState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(WaitCommandState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onSpeechToTextTranscriptReceived(const perception_msgs::msg::Transcript::SharedPtr& msg) override;
    void onStateTimeout() override;

private:
    void switchState();
};

#endif
