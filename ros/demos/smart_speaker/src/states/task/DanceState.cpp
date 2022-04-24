#include "DanceState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

DanceState::DanceState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType)
    : State(language, stateManager, desireSet, nodeHandle),
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

    constexpr bool oneshot = true;
    m_timeoutTimer =
        m_nodeHandle.createTimer(ros::Duration(DANCE_TIMEOUT_S), &DanceState::timeoutTimerCallback, this, oneshot);
}

void DanceState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void DanceState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
