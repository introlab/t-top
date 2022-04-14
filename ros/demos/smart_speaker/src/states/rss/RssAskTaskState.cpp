#include "RssAskTaskState.h"
#include "RssWaitAnswerState.h"

#include "../StateManager.h"

#include <sstream>

using namespace std;

RssAskTaskState::RssAskTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : TalkState(language, stateManager, desireSet, nodeHandle, type_index(typeid(RssWaitAnswerState)))
{
}

string RssAskTaskState::generateEnglishText(const string& personName)
{
    stringstream ss;
    ss << "Hi " << personName << ", what can I do for you? ";
    ss << "I can tell you the current weather, the weather forecast or a story. ";
    ss << "Also, I can dance to the ambient music or play a song and dance";

    return ss.str();
}

string RssAskTaskState::generateFrenchText(const string& personName)
{
    stringstream ss;
    ss << "Bonjour " << personName << ", qu'est-ce que je peux faire pour vous? ";
    ss << "Je peux vous dire la météo actuelle, les prévisions météo ou une histoire. ";
    ss << "De plus, je peux danser sur la chanson ambiante ou sur une chanson que je fais jouer.";

    return ss.str();
}
