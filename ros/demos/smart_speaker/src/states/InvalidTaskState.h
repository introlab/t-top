#ifndef SMART_SPEAKER_STATES_INVALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_INVALID_TASK_STATE_H

#include "State.h"

#include <talk/Done.h>
#include <gesture/Done.h>

class InvalidTaskState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_talkDoneSubscriber;
    ros::Subscriber m_gestureDoneSubscriber;

    uint64_t m_talkDesireId;
    uint64_t m_gestureDesireId;
    bool m_talkDone;
    bool m_gestureDone;

public:
    InvalidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~InvalidTaskState() override = default;

    DECLARE_NOT_COPYABLE(InvalidTaskState);
    DECLARE_NOT_MOVABLE(InvalidTaskState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText();

    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
    void gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg);
    void switchState();
};

inline std::type_index InvalidTaskState::type() const
{
    return std::type_index(typeid(InvalidTaskState));
}

#endif
