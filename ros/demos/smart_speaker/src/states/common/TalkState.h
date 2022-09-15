#ifndef SMART_SPEAKER_STATES_COMMON_TALK_STATE_H
#define SMART_SPEAKER_STATES_COMMON_TALK_STATE_H

#include "../State.h"

class TalkState : public State, public DesireSetObserver
{
    std::type_index m_nextStateType;

    uint64_t m_talkDesireId;

public:
    TalkState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~TalkState() override;

    DECLARE_NOT_COPYABLE(TalkState);
    DECLARE_NOT_MOVABLE(TalkState);

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

protected:
    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual std::string generateEnglishText(const std::string& parameter) = 0;
    virtual std::string generateFrenchText(const std::string& parameter) = 0;

private:
    std::string generateText(const std::string& parameter);
};

#endif
