#include "states/StateManager.h"

#include "states/rss/RssIdleState.h"
#include "states/rss/RssWaitPersonIdentificationState.h"
#include "states/rss/RssAskTaskState.h"
#include "states/rss/RssWaitAnswerState.h"
#include "states/rss/RssValidTaskState.h"
#include "states/common/InvalidTaskState.h"

#include "states/task/CurrentWeatherState.h"
#include "states/task/WeatherForecastState.h"
#include "states/rss/RssStoryState.h"
#include "states/task/DanceState.h"
#include "states/task/DancePlayedSongState.h"

#include "states/common/AfterTaskDelayState.h"

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

void startNode(
    Language language,
    ros::NodeHandle& nodeHandle,
    const string& englishStoryPath,
    const string& frenchStoryPath,
    const string& songPath,
    bool useAfterTaskDelayDurationTopic,
    const ros::Duration& afterTaskDelayDuration)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createSlowVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, nodeHandle));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(nodeHandle);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    StateManager stateManager;
    type_index idleStateType(typeid(RssIdleState));
    type_index afterTaskDelayStateType(typeid(AfterTaskDelayState));

    stateManager.addState(make_unique<RssIdleState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssWaitPersonIdentificationState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssAskTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssWaitAnswerState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssValidTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<InvalidTaskState>(language, stateManager, desireSet, nodeHandle, idleStateType));

    stateManager.addState(
        make_unique<CurrentWeatherState>(language, stateManager, desireSet, nodeHandle, afterTaskDelayStateType));
    stateManager.addState(
        make_unique<WeatherForecastState>(language, stateManager, desireSet, nodeHandle, afterTaskDelayStateType));
    stateManager.addState(
        make_unique<RssStoryState>(language, stateManager, desireSet, nodeHandle, englishStoryPath, frenchStoryPath));
    stateManager.addState(
        make_unique<DanceState>(language, stateManager, desireSet, nodeHandle, afterTaskDelayStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        afterTaskDelayStateType,
        vector<string>{songPath}));

    stateManager.addState(make_unique<AfterTaskDelayState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        idleStateType,
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDuration));

    stateManager.switchTo<RssIdleState>();

    ros::spin();
}

int startNode(int argc, char** argv)
{
    ros::init(argc, argv, "smart_speaker_rss_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    string languageString;
    Language language;
    privateNodeHandle.param<std::string>("language", languageString, "");
    if (!languageFromString(languageString, language))
    {
        ROS_ERROR("Language must be English (language=en) or French (language=fr).");
        return -1;
    }

    string englishStoryPath;
    privateNodeHandle.param<std::string>("story_path_en", englishStoryPath, "");
    if (englishStoryPath == "")
    {
        ROS_ERROR("A valid path must be set for the English story.");
        return -1;
    }

    string frenchStoryPath;
    privateNodeHandle.param<std::string>("story_path_fr", frenchStoryPath, "");
    if (frenchStoryPath == "")
    {
        ROS_ERROR("A valid path must be set for the French story.");
        return -1;
    }

    string songPath;
    privateNodeHandle.param<std::string>("song_path", songPath, "");
    if (songPath == "")
    {
        ROS_ERROR("A valid path must be set for the song.");
        return -1;
    }

    bool useAfterTaskDelayDurationTopic = false;
    privateNodeHandle.param("use_after_task_delay_duration_topic", useAfterTaskDelayDurationTopic, false);

    double afterTaskDelayDurationS;
    privateNodeHandle.param("after_task_delay_duration_s", afterTaskDelayDurationS, 0.0);
    ros::Duration afterTaskDelayDuration(afterTaskDelayDurationS);

    startNode(
        language,
        nodeHandle,
        englishStoryPath,
        frenchStoryPath,
        songPath,
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDuration);

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        return startNode(argc, argv);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Smart speaker crashed (" << e.what() << ")");
        return -1;
    }
}
