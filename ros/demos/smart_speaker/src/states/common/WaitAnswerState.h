#ifndef SMART_SPEAKER_STATES_COMMON_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_COMMON_WAIT_ANSWER_STATE_H

#include "../State.h"

#include <std_msgs/String.h>

class WaitAnswerState : public State
{
    ros::Subscriber m_speechToTextSubscriber;
    ros::Timer m_timeoutTimer;

public:
    WaitAnswerState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~WaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(WaitAnswerState);
    DECLARE_NOT_MOVABLE(WaitAnswerState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual void switchStateAfterTranscriptReceived(const std::string& text) = 0;
    virtual void switchStateAfterTimeout() = 0;

private:
    void speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index WaitAnswerState::type() const
{
    return std::type_index(typeid(WaitAnswerState));
}

#endif
