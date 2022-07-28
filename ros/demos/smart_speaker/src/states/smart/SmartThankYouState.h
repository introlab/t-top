#ifndef SMART_SPEAKER_STATES_SMART_SMART_THANK_YOU_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_THANK_YOU_STATE_H

#include "../common/TalkState.h"

#include <talk/Done.h>

#include <string>
#include <vector>

class SmartThankYouState : public TalkState
{
public:
    SmartThankYouState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SmartThankYouState() override = default;

    DECLARE_NOT_COPYABLE(SmartThankYouState);
    DECLARE_NOT_MOVABLE(SmartThankYouState);

protected:
    std::type_index type() const override;

    std::string generateEnglishText(const std::string& _) override;
    std::string generateFrenchText(const std::string& _) override;
};

inline std::type_index SmartThankYouState::type() const
{
    return std::type_index(typeid(SmartThankYouState));
}

#endif
