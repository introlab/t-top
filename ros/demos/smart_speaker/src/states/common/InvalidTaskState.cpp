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
      m_gestureDesireId(MAX_DESIRE_ID)
{
    m_desireSet->addObserver(this);
}

InvalidTaskState::~InvalidTaskState()
{
    m_desireSet->removeObserver(this);
}

void InvalidTaskState::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    if (!enabled() || m_desireSet->contains(m_talkDesireId) || m_desireSet->contains(m_gestureDesireId))
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}

void InvalidTaskState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

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
