#ifndef SMART_SPEAKER_STATES_COMMON_TALK_STATE_H
#define SMART_SPEAKER_STATES_COMMON_TALK_STATE_H

#include "../State.h"

#include <talk/Done.h>

class TalkState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    TalkState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~TalkState() override = default;

    DECLARE_NOT_COPYABLE(TalkState);
    DECLARE_NOT_MOVABLE(TalkState);

protected:
    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual std::string generateEnglishText(const std::string& personName) = 0;
    virtual std::string generateFrenchText(const std::string& personName) = 0;

private:
    std::string generateText(const std::string& personName);
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

#endif
