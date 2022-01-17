#include "states/StateManager.h"

#include "states/IdleState.h"
#include "states/WaitPersonIdentificationState.h"
#include "states/AskTaskState.h"
#include "states/WaitAnswerState.h"
#include "states/ValidTaskState.h"
#include "states/InvalidTaskState.h"

#include "states/CurrentWeatherState.h"
#include "states/WeatherForecastState.h"
#include "states/StoryState.h"
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

void startNode(ros::NodeHandle& nodeHandle,
    const string& language,
    const string& storyPath,
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
    stateManager.addState(make_unique<IdleState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<WaitPersonIdentificationState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<AskTaskState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<WaitAnswerState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<ValidTaskState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<InvalidTaskState>(stateManager, desireSet, nodeHandle));

    stateManager.addState(make_unique<CurrentWeatherState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<WeatherForecastState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<StoryState>(stateManager, desireSet, nodeHandle, storyPath));
    stateManager.addState(make_unique<DanceState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<DancePlayedSongState>(stateManager, desireSet, nodeHandle, songPath));

    stateManager.switchTo<IdleState>();

    ros::spin();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "smart_speaker_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    string language;
    privateNodeHandle.param<std::string>("language", language, "");
    if (language != "en")
    {
        ROS_ERROR("Language must be English (language=en).");
        return -1;
    }

    string storyPath;
    privateNodeHandle.param<std::string>("story_path", storyPath, "");
    if (storyPath == "")
    {
        ROS_ERROR("A valid path must be set for the story.");
        return -1;
    }

    string songPath;
    privateNodeHandle.param<std::string>("song_path", songPath, "");
    if (songPath == "")
    {
        ROS_ERROR("A valid path must be set for the song.");
        return -1;
    }

    startNode(nodeHandle, language, storyPath, songPath);

    return 0;
}
