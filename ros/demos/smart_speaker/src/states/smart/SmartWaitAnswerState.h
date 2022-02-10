#ifndef SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_WAIT_ANSWER_STATE_H

#include "../WaitAnswerState.h"

#include <std_msgs/String.h>

#include <string>
#include <random>

class SmartWaitAnswerState : public WaitAnswerState
{
    std::vector<std::string> m_songNames;

    std::string m_weatherWord;
    std::string m_danceWord;

    std::mt19937 m_randomGenerator;
    std::uniform_int_distribution<size_t> m_songIndexDistribution;

public:
    SmartWaitAnswerState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        vector<string> songNames);
    ~SmartWaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(SmartWaitAnswerState);
    DECLARE_NOT_MOVABLE(SmartWaitAnswerState);

protected:
    std::type_index type() const override;

    void switchStateAfterTranscriptReceived(const std::string& text) override;
    void switchStateAfterTimeout() override;

private:
    size_t getSongIndex(const std::string& text);
};

inline std::type_index SmartWaitAnswerState::type() const
{
    return std::type_index(typeid(SmartWaitAnswerState));
}

#endif
