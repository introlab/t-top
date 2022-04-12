#include "AfterTaskDelayState.h"

#include "../StateManager.h"

using namespace std;

AfterTaskDelayState::AfterTaskDelayState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType,
    bool useAfterTaskDelayDurationTopic,
    ros::Duration duration)
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_useAfterTaskDelayDurationTopic(useAfterTaskDelayDurationTopic),
      m_duration(duration)
{
    if (m_useAfterTaskDelayDurationTopic)
    {
        m_readySubscriber =
            nodeHandle.subscribe("after_task_delay_state_ready", 1, &AfterTaskDelayState::readyCallback, this);
    }
}

void AfterTaskDelayState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    if (!m_useAfterTaskDelayDurationTopic)
    {
        constexpr bool oneshot = true;
        m_timeoutTimer =
            m_nodeHandle.createTimer(m_duration, &AfterTaskDelayState::timeoutTimerCallback, this, oneshot);
    }
}

void AfterTaskDelayState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void AfterTaskDelayState::readyCallback(const std_msgs::Empty::ConstPtr& msg)
{
    if (!enabled() || !m_useAfterTaskDelayDurationTopic)
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}

void AfterTaskDelayState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
