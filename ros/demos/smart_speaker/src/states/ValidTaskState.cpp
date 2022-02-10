#include "ValidTaskState.h"
#include "StateManager.h"

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

ValidTaskState::ValidTaskState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(language, stateManager, desireSet, nodeHandle),
        m_talkDesireId(MAX_DESIRE_ID),
        m_gestureDesireId(MAX_DESIRE_ID),
        m_talkDone(false),
        m_gestureDone(false)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1,
        &ValidTaskState::talkDoneSubscriberCallback, this);
    m_gestureDoneSubscriber = nodeHandle.subscribe("gesture/done", 1,
        &ValidTaskState::gestureDoneSubscriberCallback, this);
}

void ValidTaskState::enable(const string& parameter)
{
    State::enable(parameter);

    m_task = parameter;
    m_talkDone = false;
    m_gestureDone = false;

    auto gestureDesire = make_unique<GestureDesire>("yes");
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("happy");
    auto talkDesire = make_unique<TalkDesire>(generateText());
    m_talkDesireId = talkDesire->id();
    m_gestureDesireId = gestureDesire->id();

    m_desireIds.emplace_back(gestureDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());
    m_desireIds.emplace_back(talkDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(gestureDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
    m_desireSet->addDesire(move(talkDesire));
}

void ValidTaskState::disable()
{
    State::disable();

    m_talkDesireId = MAX_DESIRE_ID;
    m_gestureDesireId = MAX_DESIRE_ID;
}

string ValidTaskState::generateText()
{
    switch (language())
    {
    case Language::ENGLISH:
        return "Yes, of course.";
    case Language::FRENCH:
        return "Oui, bien sûr.";
    }

    return "";
}

void ValidTaskState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_talkDone = true;
    if (m_talkDone && m_gestureDone)
    {
        switchState(m_task);
    }
}

void ValidTaskState::gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_gestureDesireId)
    {
        return;
    }

    m_gestureDone = true;
    if (m_talkDone && m_gestureDone)
    {
        switchState(m_task);
    }
}
