#ifndef SMART_SPEAKER_STATES_COMMON_VALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_COMMON_VALID_TASK_STATE_H

#include "../State.h"

class ValidTaskState : public State, public DesireSetObserver
{
    std::string m_task;

    uint64_t m_talkDesireId;
    uint64_t m_gestureDesireId;

public:
    ValidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~ValidTaskState() override;

    DECLARE_NOT_COPYABLE(ValidTaskState);
    DECLARE_NOT_MOVABLE(ValidTaskState);

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

protected:
    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

    virtual void switchState(const std::string& task) = 0;

private:
    std::string generateText();
};

#endif
