#ifndef SMART_SPEAKER_STATES_COMMON_ASK_TASK_STATE_H
#define SMART_SPEAKER_STATES_COMMON_ASK_TASK_STATE_H

#include "../State.h"

#include <talk/Done.h>

class AskTaskState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    AskTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~AskTaskState() override = default;

    DECLARE_NOT_COPYABLE(AskTaskState);
    DECLARE_NOT_MOVABLE(AskTaskState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual std::string generateEnglishText(const std::string& personName) = 0;
    virtual std::string generateFrenchText(const std::string& personName) = 0;

private:
    std::string generateText(const std::string& personName);
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index AskTaskState::type() const
{
    return std::type_index(typeid(AskTaskState));
}

#endif
