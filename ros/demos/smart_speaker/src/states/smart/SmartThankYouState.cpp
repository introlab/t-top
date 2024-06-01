#include "SmartThankYouState.h"

#include "../StateManager.h"

#include "../common/AfterTaskDelayState.h"

#include "../../StringUtils.h"

using namespace std;

SmartThankYouState::SmartThankYouState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node)
    : TalkState(language, stateManager, desireSet, move(node), type_index(typeid(AfterTaskDelayState)))
{
}

string SmartThankYouState::generateEnglishText(const string& _)
{
    return "Thank you for participating.";
}

string SmartThankYouState::generateFrenchText(const string& _)
{
    return "Merci d'avoir participer.";
}
