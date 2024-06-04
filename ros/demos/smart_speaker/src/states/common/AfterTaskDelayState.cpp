#include "AfterTaskDelayState.h"

#include "../StateManager.h"

using namespace std;

AfterTaskDelayState::AfterTaskDelayState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    type_index nextStateType,
    bool useAfterTaskDelayDurationTopic,
    std::chrono::milliseconds durationMs)
    : State(language, stateManager, desireSet, move(node)),
      m_nextStateType(nextStateType),
      m_useAfterTaskDelayDurationTopic(useAfterTaskDelayDurationTopic),
      m_durationMs(durationMs)
{
    if (m_useAfterTaskDelayDurationTopic)
    {
        m_startButtonSubscriber = m_node->create_subscription<std_msgs::msg::Empty>(
            "daemon/start_button_pressed",
            1,
            [this](const std_msgs::msg::Empty::SharedPtr msg) { startButtonCallback(msg); });
    }
}

void AfterTaskDelayState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    if (!m_useAfterTaskDelayDurationTopic)
    {
        m_timeoutTimer = m_node->create_wall_timer(m_durationMs, [this]() { timeoutTimerCallback(); });
    }
}

void AfterTaskDelayState::disable()
{
    State::disable();

    if (m_timeoutTimer)
    {
        m_timeoutTimer->cancel();
        m_timeoutTimer = nullptr;
    }
}

void AfterTaskDelayState::startButtonCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    if (!enabled() || !m_useAfterTaskDelayDurationTopic)
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}

void AfterTaskDelayState::timeoutTimerCallback()
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}
