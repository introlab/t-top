#ifndef SMART_SPEAKER_STATES_RSS_RSS_ASK_TASK_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_ASK_TASK_STATE_H

#include "../State.h"

#include <talk/Done.h>

class RssAskTaskState : public State
{
    ros::Subscriber m_talkDoneSubscriber;

    uint64_t m_talkDesireId;

public:
    RssAskTaskState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssAskTaskState() override = default;

    DECLARE_NOT_COPYABLE(RssAskTaskState);
    DECLARE_NOT_MOVABLE(RssAskTaskState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    std::string generateText(const std::string& personName);
    std::string generateEnglishText(const std::string& personName);
    std::string generateFrenchText(const std::string& personName);
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index RssAskTaskState::type() const
{
    return std::type_index(typeid(RssAskTaskState));
}

#endif
