#include "SmartValidTaskState.h"
#include "SmartIdleState.h"

#include "../StateManager.h"
#include "../CurrentWeatherState.h"
#include "../DancePlayedSongState.h"

#include "../../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

#include <sstream>

using namespace std;

SmartValidTaskState::SmartValidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : ValidTaskState(language, stateManager, desireSet, nodeHandle)
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
        ROS_ERROR_STREAM("Invalid task (" << task << ")");
        m_stateManager.switchTo<SmartIdleState>();
    }
}
