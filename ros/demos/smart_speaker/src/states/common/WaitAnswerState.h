#ifndef SMART_SPEAKER_STATES_COMMON_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_COMMON_WAIT_ANSWER_STATE_H

#include "../State.h"

#include <speech_to_text/Transcript.h>

class WaitAnswerState : public State
{
    ros::Subscriber m_speechToTextSubscriber;
    ros::Timer m_timeoutTimer;

    bool m_transcriptReceived;

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
    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual void switchStateAfterTranscriptReceived(const std::string& text, bool isFinal) = 0;
    virtual void switchStateAfterTimeout(bool transcriptReceived) = 0;

private:
    void speechToTextSubscriberCallback(const speech_to_text::Transcript::ConstPtr& msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

#endif
