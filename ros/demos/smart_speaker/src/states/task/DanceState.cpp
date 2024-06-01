#include "DanceState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

DanceState::DanceState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    type_index nextStateType)
    : State(language, stateManager, desireSet, move(node)),
      m_nextStateType(nextStateType)
{
}

void DanceState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto danceDesire = make_unique<DanceDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(danceDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(danceDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));

    m_timeoutTimer =
        m_node->create_wall_timer(chrono::seconds(DANCE_TIMEOUT_S), [this]() { timeoutTimerCallback(); });
}

void DanceState::disable()
{
    State::disable();

    if (m_timeoutTimer)
    {
        m_timeoutTimer->cancel();
        m_timeoutTimer = nullptr;
    }
}

void DanceState::timeoutTimerCallback()
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
