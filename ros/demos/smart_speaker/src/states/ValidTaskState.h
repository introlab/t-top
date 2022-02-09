#ifndef SMART_SPEAKER_STATES_VALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_VALID_TASK_STATE_H

#include "State.h"

#include <talk/Done.h>
#include <gesture/Done.h>

class ValidTaskState : public State
{
    ros::Subscriber m_talkDoneSubscriber;
    ros::Subscriber m_gestureDoneSubscriber;

    std::string m_task;

    uint64_t m_talkDesireId;
    uint64_t m_gestureDesireId;
    bool m_talkDone;
    bool m_gestureDone;

public:
    ValidTaskState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~ValidTaskState() override = default;

    DECLARE_NOT_COPYABLE(ValidTaskState);
    DECLARE_NOT_MOVABLE(ValidTaskState);

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

    virtual void switchState(const std::string& task) = 0;

private:
    std::string generateText();

    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
    void gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg);
};

#endif
