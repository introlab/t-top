#include "TalkState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

TalkState::TalkState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType)
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_talkDesireId(MAX_DESIRE_ID)
{
    m_desireSet->addObserver(this);
}

TalkState::~TalkState()
{
    m_desireSet->removeObserver(this);
}

void TalkState::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    if (!enabled() || m_desireSet->contains(m_talkDesireId))
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}

void TalkState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto faceFollowingDesire = make_unique<NearestFaceFollowingDesire>();
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

void TalkState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
}

string TalkState::generateText(const string& parameter)
{
    switch (language())
    {
        case Language::ENGLISH:
            return generateEnglishText(parameter);
        case Language::FRENCH:
            return generateFrenchText(parameter);
    }

    return "";
}
