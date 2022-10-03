#ifndef SMART_SPEAKER_STATES_COMMON_INVALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_COMMON_INVALID_TASK_STATE_H

#include "../State.h"

class InvalidTaskState : public State, public DesireSetObserver
{
    std::type_index m_nextStateType;

    uint64_t m_talkDesireId;
    uint64_t m_gestureDesireId;

public:
    InvalidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType);
    ~InvalidTaskState() override;

    DECLARE_NOT_COPYABLE(InvalidTaskState);
    DECLARE_NOT_MOVABLE(InvalidTaskState);

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual std::string generateText();
};

inline std::type_index InvalidTaskState::type() const
{
    return std::type_index(typeid(InvalidTaskState));
}

#endif
