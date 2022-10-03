#include "ValidTaskState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

ValidTaskState::ValidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : State(language, stateManager, desireSet, nodeHandle),
      m_talkDesireId(MAX_DESIRE_ID),
      m_gestureDesireId(MAX_DESIRE_ID)
{
    m_desireSet->addObserver(this);
}

ValidTaskState::~ValidTaskState()
{
    m_desireSet->removeObserver(this);
}

void ValidTaskState::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    if (!enabled() || m_desireSet->contains(m_talkDesireId) || m_desireSet->contains(m_gestureDesireId))
    {
        return;
    }

    switchState(m_task);
}

void ValidTaskState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    m_task = parameter;

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
