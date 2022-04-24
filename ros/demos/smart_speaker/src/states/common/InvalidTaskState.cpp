#include "InvalidTaskState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

InvalidTaskState::InvalidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType)
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_talkDesireId(MAX_DESIRE_ID),
      m_gestureDesireId(MAX_DESIRE_ID),
      m_talkDone(false),
      m_gestureDone(false)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1, &InvalidTaskState::talkDoneSubscriberCallback, this);
    m_gestureDoneSubscriber =
        nodeHandle.subscribe("gesture/done", 1, &InvalidTaskState::gestureDoneSubscriberCallback, this);
}

void InvalidTaskState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    m_talkDone = false;
    m_gestureDone = false;

    auto gestureDesire = make_unique<GestureDesire>("no");
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("sad");
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

void InvalidTaskState::disable()
{
    State::disable();

    m_talkDesireId = MAX_DESIRE_ID;
    m_gestureDesireId = MAX_DESIRE_ID;
}

string InvalidTaskState::generateText()
{
    switch (language())
    {
        case Language::ENGLISH:
            return "I cannot do that.";
        case Language::FRENCH:
            return "Je ne peux pas faire cela.";
    }

    return "";
}

void InvalidTaskState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
    {
        return;
    }

    m_talkDone = true;
    switchState();
}

void InvalidTaskState::gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_gestureDesireId)
    {
        return;
    }

    m_gestureDone = true;
    switchState();
}

void InvalidTaskState::switchState()
{
    if (m_talkDone && m_gestureDone)
    {
        m_stateManager.switchTo(m_nextStateType);
    }
}
