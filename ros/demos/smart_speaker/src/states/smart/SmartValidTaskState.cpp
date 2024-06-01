#include "SmartValidTaskState.h"
#include "SmartIdleState.h"

#include "../StateManager.h"

#include "../task/CurrentWeatherState.h"
#include "../task/DancePlayedSongState.h"

#include "../../StringUtils.h"

using namespace std;

SmartValidTaskState::SmartValidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node)
    : ValidTaskState(language, stateManager, desireSet, move(node))
{
}

void SmartValidTaskState::switchState(const string& task)
{
    auto splitTask = splitStrings(task, "|");
    if (splitTask.size() == 1 && splitTask[0] == CURRENT_WEATHER_TASK)
    {
        m_stateManager.switchTo<CurrentWeatherState>();
    }
    else if (splitTask.size() == 2 && splitTask[0] == DANCE_TASK)
    {
        m_stateManager.switchTo<DancePlayedSongState>(splitTask[1]);
    }
    else
    {
        RCLCPP_ERROR_STREAM(m_node->get_logger(), "Invalid task (" << task << ")");
        m_stateManager.switchTo<SmartIdleState>();
    }
}
