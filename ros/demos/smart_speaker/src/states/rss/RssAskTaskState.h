#ifndef SMART_SPEAKER_STATES_RSS_RSS_ASK_TASK_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_ASK_TASK_STATE_H

#include "../common/TalkState.h"

class RssAskTaskState : public TalkState
{
public:
    RssAskTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssAskTaskState() override = default;

    DECLARE_NOT_COPYABLE(RssAskTaskState);
    DECLARE_NOT_MOVABLE(RssAskTaskState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& personName) override;
    std::string generateFrenchText(const std::string& personName) override;
};

inline std::type_index RssAskTaskState::type() const
{
    return std::type_index(typeid(RssAskTaskState));
}

#endif
