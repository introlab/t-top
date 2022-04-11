#ifndef SMART_SPEAKER_STATES_INVALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_INVALID_TASK_STATE_H

#include "../common/InvalidTaskState.h"

#include <talk/Done.h>
#include <gesture/Done.h>

class SmartPeopleInvalidTaskState : public InvalidTaskState
{
public:
    SmartPeopleInvalidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SmartPeopleInvalidTaskState() override = default;

    DECLARE_NOT_COPYABLE(SmartPeopleInvalidTaskState);
    DECLARE_NOT_MOVABLE(SmartPeopleInvalidTaskState);

protected:
    std::type_index type() const override;

    std::string generateText() override;

private:
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
    void gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg);
    void switchState();
};

inline std::type_index SmartPeopleInvalidTaskState::type() const
{
    return std::type_index(typeid(SmartPeopleInvalidTaskState));
}

#endif
