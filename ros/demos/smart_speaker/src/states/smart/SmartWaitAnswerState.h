#ifndef SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H

#include "../common/WaitAnswerState.h"

#include <std_msgs/String.h>

#include <string>
#include <random>

class SmartWaitAnswerState : public WaitAnswerState
{
    bool m_singleTaskPerPerson;
    size_t m_taskCount;

    std::vector<std::vector<std::string>> m_songKeywords;

    std::string m_nothingWord;
    std::string m_weatherWord;
    std::string m_danceWord;

    std::mt19937 m_randomGenerator;
    std::uniform_int_distribution<size_t> m_songIndexDistribution;

public:
    SmartWaitAnswerState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        bool singleTaskPerPerson,
        std::vector<std::vector<std::string>> songKeywords);
    ~SmartWaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(SmartWaitAnswerState);
    DECLARE_NOT_MOVABLE(SmartWaitAnswerState);

protected:
    std::type_index type() const override;

    void switchStateAfterTranscriptReceived(const std::string& text) override;
    void switchStateAfterTimeout() override;

private:
    size_t getSongIndex(const std::string& text);
    bool containsAllKeywords(const std::string& text, const std::vector<std::string>& keywords);
};

inline std::type_index SmartWaitAnswerState::type() const
{
    return std::type_index(typeid(SmartWaitAnswerState));
}

#endif
