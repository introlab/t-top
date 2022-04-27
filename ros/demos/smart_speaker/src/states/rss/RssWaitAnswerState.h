#ifndef SMART_SPEAKER_STATES_RSS_RSS_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_WAIT_ANSWER_STATE_H

#include "../common/WaitAnswerState.h"

#include <std_msgs/String.h>

class RssWaitAnswerState : public WaitAnswerState
{
    std::string m_weatherWord;
    std::string m_forecastWord;
    std::string m_storyWord;
    std::string m_danceWord;
    std::string m_songWord;

public:
    RssWaitAnswerState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssWaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(RssWaitAnswerState);
    DECLARE_NOT_MOVABLE(RssWaitAnswerState);

protected:
    std::type_index type() const override;

    void switchStateAfterTranscriptReceived(const std::string& text, bool isFinal) override;
    void switchStateAfterTimeout(bool transcriptReceived) override;
};

inline std::type_index RssWaitAnswerState::type() const
{
    return std::type_index(typeid(RssWaitAnswerState));
}

#endif
