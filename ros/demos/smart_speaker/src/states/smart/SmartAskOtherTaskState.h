#ifndef SMART_SPEAKER_STATES_SMART_SAMRT_ASK_TASK_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_ASK_TASK_STATE_H

#include "../common/AskTaskState.h"

#include <talk/Done.h>

#include <string>
#include <vector>

class SmartAskOtherTaskState : public AskTaskState
{
public:
    SmartAskOtherTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SmartAskOtherTaskState() override = default;

    DECLARE_NOT_COPYABLE(SmartAskOtherTaskState);
    DECLARE_NOT_MOVABLE(SmartAskOtherTaskState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& personName) override;
    std::string generateFrenchText(const std::string& personName) override;
};

inline std::type_index SmartAskOtherTaskState::type() const
{
    return std::type_index(typeid(SmartAskOtherTaskState));
}

#endif
