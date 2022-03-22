#include "SmartAskTaskState.h"
#include "SmartWaitAnswerState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

SmartAskTaskState::SmartAskTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : AskTaskState(language, stateManager, desireSet, nodeHandle, type_index(typeid(SmartWaitAnswerState)))
{
}

string SmartAskTaskState::generateEnglishText(const string& personName)
{
    stringstream ss;
    ss << "Hi " << personName << ", what can I do for you?";

    return ss.str();
}

string SmartAskTaskState::generateFrenchText(const string& personName)
{
    stringstream ss;
    ss << "Bonjour " << personName << ", qu'est-ce que je peux faire pour vous?";

    return ss.str();
}
