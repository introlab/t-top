#ifndef SMART_SPEAKER_STATES_SMART_SAMRT_ASK_TASK_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_ASK_TASK_STATE_H

#include "../AskTaskState.h"

#include <talk/Done.h>

class SmartAskTaskState : public AskTaskState
{
public:
    SmartAskTaskState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SmartAskTaskState() override = default;

    DECLARE_NOT_COPYABLE(SmartAskTaskState);
    DECLARE_NOT_MOVABLE(SmartAskTaskState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& personName);
    std::string generateFrenchText(const std::string& personName);
};

inline std::type_index SmartAskTaskState::type() const
{
    return std::type_index(typeid(SmartAskTaskState));
}

#endif
