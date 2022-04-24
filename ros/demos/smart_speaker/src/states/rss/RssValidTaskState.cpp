#include "RssValidTaskState.h"
#include "RssIdleState.h"
#include "RssStoryState.h"

#include "../StateManager.h"

#include "../task/CurrentWeatherState.h"
#include "../task/WeatherForecastState.h"
#include "../task/DanceState.h"
#include "../task/DancePlayedSongState.h"

using namespace std;

RssValidTaskState::RssValidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : ValidTaskState(language, stateManager, desireSet, nodeHandle)
{
}

void RssValidTaskState::switchState(const string& task)
{
    if (task == CURRENT_WEATHER_TASK)
    {
        m_stateManager.switchTo<CurrentWeatherState>();
    }
    else if (task == WEATHER_FORECAST_TASK)
    {
        m_stateManager.switchTo<WeatherForecastState>();
    }
    else if (task == STORY_TASK)
    {
        m_stateManager.switchTo<RssStoryState>();
    }
    else if (task == DANCE_TASK)
    {
        m_stateManager.switchTo<DanceState>();
    }
    else if (task == DANCE_PLAYED_SONG_TASK)
    {
        m_stateManager.switchTo<DancePlayedSongState>("0");  // First song
    }
    else
    {
        ROS_ERROR_STREAM("Invalid task (" << task << ")");
        m_stateManager.switchTo<RssIdleState>();
    }
}
