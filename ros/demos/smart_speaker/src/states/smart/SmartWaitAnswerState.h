#ifndef SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H

#include "../common/WaitAnswerState.h"

#include <std_msgs/String.h>

#include <string>

class SmartWaitAnswerState : public WaitAnswerState
{
    std::vector<std::vector<std::string>> m_songKeywords;

    std::string m_nothingWord;
    std::string m_weatherWord;
    std::string m_danceWord;

public:
    SmartWaitAnswerState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::vector<std::vector<std::string>> songKeywords);
    ~SmartWaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(SmartWaitAnswerState);
    DECLARE_NOT_MOVABLE(SmartWaitAnswerState);

protected:
    std::type_index type() const override;

    void switchStateAfterTranscriptReceived(const std::string& text, bool isFinal) override;
    void switchStateAfterTimeout(bool transcriptReceived) override;

private:
    size_t getSongIndex(const std::string& text);
    bool containsAllKeywords(const std::string& text, const std::vector<std::string>& keywords);
};

inline std::type_index SmartWaitAnswerState::type() const
{
    return std::type_index(typeid(SmartWaitAnswerState));
}

#endif
