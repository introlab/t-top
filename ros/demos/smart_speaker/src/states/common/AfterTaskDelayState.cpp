#include "AfterTaskDelayState.h"

#include "../StateManager.h"

using namespace std;

AfterTaskDelayState::AfterTaskDelayState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType,
    ros::Duration duration)
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_duration(duration)
{
}

void AfterTaskDelayState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    constexpr bool oneshot = true;
    m_timeoutTimer = m_nodeHandle.createTimer(m_duration, &AfterTaskDelayState::timeoutTimerCallback, this, oneshot);
}

void AfterTaskDelayState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void AfterTaskDelayState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
