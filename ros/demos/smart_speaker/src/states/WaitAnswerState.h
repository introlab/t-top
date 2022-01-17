#ifndef SMART_SPEAKER_STATES_WAIT_ANSWER_STATE_H
#define SMART_SPEAKER_STATES_WAIT_ANSWER_STATE_H

#include "State.h"

#include <std_msgs/String.h>

class WaitAnswerState : public State
{
    ros::Subscriber m_speechToTextSubscriber;
    ros::Timer m_timeoutTimer;

    std::string m_weatherWord;
    std::string m_forecastWord;
    std::string m_storyWord;
    std::string m_danceWord;
    std::string m_songWord;

public:
    WaitAnswerState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~WaitAnswerState() override = default;

    DECLARE_NOT_COPYABLE(WaitAnswerState);
    DECLARE_NOT_MOVABLE(WaitAnswerState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index WaitAnswerState::type() const
{
    return std::type_index(typeid(WaitAnswerState));
}

#endif
