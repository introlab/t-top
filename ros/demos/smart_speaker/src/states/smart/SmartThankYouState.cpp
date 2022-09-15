#include "SmartThankYouState.h"

#include "../StateManager.h"

#include "../common/AfterTaskDelayState.h"

#include "../../StringUtils.h"

using namespace std;

SmartThankYouState::SmartThankYouState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : TalkState(language, stateManager, desireSet, nodeHandle, type_index(typeid(AfterTaskDelayState)))
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
