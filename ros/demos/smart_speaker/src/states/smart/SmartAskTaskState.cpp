#include "SmartAskTaskState.h"
#include "SmartWaitAnswerState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <sstream>

using namespace std;

SmartAskTaskState::SmartAskTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    vector<string> songNames)
    : TalkState(language, stateManager, desireSet, nodeHandle, type_index(typeid(SmartWaitAnswerState))),
      m_songNames(move(songNames))
{
}

string SmartAskTaskState::generateEnglishText(const string& personName)
{
    stringstream ss;
    ss << "Hi " << personName << ", what can I do for you?";
    ss << "I can tell you the weather or dance. ";
    ss << "The available songs are " << mergeNames(m_songNames, getAndWord()) << ".";

    return ss.str();
}

string SmartAskTaskState::generateFrenchText(const string& personName)
{
    stringstream ss;
    ss << "Bonjour " << personName << ", qu'est-ce que je peux faire pour vous?";
    ss << "Je peux vous dire la météo ou danser. ";
    ss << "Les chansons disponibles sont " << mergeNames(m_songNames, getAndWord()) << ".";

    return ss.str();
}
