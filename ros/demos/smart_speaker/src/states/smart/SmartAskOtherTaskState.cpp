#include "SmartAskOtherTaskState.h"
#include "SmartWaitAnswerState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

using namespace std;

SmartAskOtherTaskState::SmartAskOtherTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : AskTaskState(language, stateManager, desireSet, nodeHandle, type_index(typeid(SmartWaitAnswerState)))
{
}

string SmartAskOtherTaskState::generateEnglishText(const string& personName)
{
    return "What else can I do for you?";
}

string SmartAskOtherTaskState::generateFrenchText(const string& personName)
{
    return "Qu'est-ce que je peux faire d'autre pour vous?";
}
