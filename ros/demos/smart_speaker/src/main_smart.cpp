#include "states/StateManager.h"

#include "states/smart/SmartIdleState.h"
#include "states/smart/SmartAskTaskState.h"
#include "states/smart/SmartWaitAnswerState.h"
#include "states/smart/SmartValidTaskState.h"
#include "states/InvalidTaskState.h"

#include "states/CurrentWeatherState.h"
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
    const vector<string>& songNames,
    const vector<string>& songPaths)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createFastVideoAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, nodeHandle));

    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"motor", 1}, {"sound", 1}}, move(solver));

    StateManager stateManager;
    type_index idleStateType(typeid(SmartIdleState));

    stateManager.addState(make_unique<SmartIdleState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<SmartAskTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<SmartWaitAnswerState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<SmartValidTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<InvalidTaskState>(language, stateManager, desireSet, nodeHandle, idleStateType));

    stateManager.addState(make_unique<CurrentWeatherState>(language, stateManager, desireSet, nodeHandle, idleStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(language, stateManager, desireSet, nodeHandle, idleStateType, vector<string>{songPath}));

    stateManager.switchTo<SmartIdleState>();

    ros::spin();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "smart_speaker_smart_node");
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

    vector<string> songNames;
    privateNodeHandle.param<std::string>("song_names", songNames, "");
    if (songNames.size())
    {
        ROS_ERROR("At least one valid path must be set for the songs.");
        return -1;
    }

    vector<string> songPaths;
    privateNodeHandle.param<std::string>("song_paths", songPaths, "");
    if (songPaths.size())
    {
        ROS_ERROR("At least one valid path must be set for the songs.");
        return -1;
    }

    if (songNames.size() != songPaths.size())
    {
        ROS_ERROR("The size of song_names and song_paths must be the same.");
        return -1;
    }

    startNode(language, nodeHandle, englishStoryPath, frenchStoryPath, songNames, songPaths);

    return 0;
}
