#include "AskTaskState.h"
#include "StateManager.h"
#include "WaitAnswerState.h"

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

AskTaskState::AskTaskState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(language, stateManager, desireSet, nodeHandle),
        m_talkDesireId(MAX_DESIRE_ID)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1,
        &AskTaskState::talkDoneSubscriberCallback, this);
}

void AskTaskState::enable(const string& parameter)
{
    State::enable(parameter);

    auto faceFollowingDesire = make_unique<FaceFollowingDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");
    auto talkDesire = make_unique<TalkDesire>(generateText(parameter));
    m_talkDesireId = talkDesire->id();

    m_desireIds.emplace_back(faceFollowingDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());
    m_desireIds.emplace_back(talkDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(faceFollowingDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
    m_desireSet->addDesire(move(talkDesire));
}

void AskTaskState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
}

string AskTaskState::generateText(const string& personName)
{
    switch (language())
    {
    case Language::ENGLISH:
        return generateEnglishText(personName);
    case Language::FRENCH:
        return generateFrenchText(personName);
    }

    return "";
}

string AskTaskState::generateEnglishText(const string& personName)
{
    stringstream ss;
    ss << "Hi " << personName << ", what can I do for you? ";
    ss << "I can tell you the current weather, the weather forecast or a story. ";
    ss << "Also, I can dance to the ambient music or play a song and dance";

    return ss.str();
}

string AskTaskState::generateFrenchText(const string& personName)
{
    stringstream ss;
    ss << "Bonjour " << personName << ", qu'est-ce que je peux faire pour vous? ";
    ss << "Je peux vous dire la météo actuelle, les prévisions météo ou une histoire. ";
    ss << "De plus, je peux danser sur la chanson ambiante ou sur une chanson que je fais jouer.";

    return ss.str();
}

void AskTaskState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_stateManager.switchTo<WaitAnswerState>();
}
