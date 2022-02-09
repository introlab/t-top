#include "states/StateManager.h"

#include "states/rss/RssIdleState.h"
#include "states/rss/RssWaitPersonIdentificationState.h"
#include "states/rss/RssAskTaskState.h"
#include "states/rss/RssWaitAnswerState.h"
#include "states/rss/RssValidTaskState.h"
#include "states/InvalidTaskState.h"

#include "states/CurrentWeatherState.h"
#include "states/WeatherForecastState.h"
#include "states/rss/RssStoryState.h"
#include "states/DanceState.h"
#include "states/DancePlayedSongState.h"

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top/hbba_lite/Strategies.h>

#include <memory>

using namespace std;

void startNode(Language language,
    ros::NodeHandle& nodeHandle,
    const string& englishStoryPath,
    const string& frenchStoryPath,
    const string& songPath)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createSlowVideoAnalyzerStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzerStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, nodeHandle));

    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"motor", 1}, {"sound", 1}}, move(solver));

    StateManager stateManager;
    type_index idleStateType(typeid(RssIdleState));

    stateManager.addState(make_unique<RssIdleState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssWaitPersonIdentificationState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssAskTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssWaitAnswerState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<RssValidTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<InvalidTaskState>(language, stateManager, desireSet, nodeHandle, idleStateType));

    stateManager.addState(make_unique<CurrentWeatherState>(language, stateManager, desireSet, nodeHandle, idleStateType));
    stateManager.addState(make_unique<WeatherForecastState>(language, stateManager, desireSet, nodeHandle, idleStateType));
    stateManager.addState(make_unique<RssStoryState>(language, stateManager, desireSet, nodeHandle, englishStoryPath, frenchStoryPath));
    stateManager.addState(make_unique<DanceState>(language, stateManager, desireSet, nodeHandle, idleStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(language, stateManager, desireSet, nodeHandle, idleStateType, songPath));

    stateManager.switchTo<RssIdleState>();

    ros::spin();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "smart_speaker_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    string languageString;
    Language language;
    privateNodeHandle.param<std::string>("language", languageString, "");
    if (languageString == "en")
    {
        language = Language::ENGLISH;
    }
    else if (languageString == "fr")
    {
        language = Language::FRENCH;
    }
    else
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

    startNode(language, nodeHandle, englishStoryPath, frenchStoryPath, songPath);

    return 0;
}
