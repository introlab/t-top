#include "states/StateManager.h"

#include "states/smart/SmartIdleState.h"
#include "states/smart/SmartAskTaskState.h"
#include "states/smart/SmartWaitAnswerState.h"
#include "states/smart/SmartValidTaskState.h"
#include "states/InvalidTaskState.h"

#include "states/CurrentWeatherState.h"
#include "states/DancePlayedSongState.h"

#include "states/AfterTaskDelayState.h"

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top/hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

void startNode(
    Language language,
    ros::NodeHandle& nodeHandle,
    double personDistanceThreshold,
    const std::string& personDistanceFrame,
    double noseConfidenceThreshold,
    size_t videoAnalysisMessageCountThreshold,
    size_t videoAnalysisMessageCountTolerance,
    const vector<string>& songNames,
    const vector<string>& songPaths,
    const ros::Duration& afterTaskDelayDuration)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

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
    type_index afterTaskDelayStateType(typeid(AfterTaskDelayState));

    stateManager.addState(make_unique<SmartIdleState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        personDistanceThreshold,
        personDistanceFrame,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance));
    stateManager.addState(make_unique<SmartAskTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<SmartWaitAnswerState>(language, stateManager, desireSet, nodeHandle, songNames));
    stateManager.addState(make_unique<SmartValidTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<InvalidTaskState>(language, stateManager, desireSet, nodeHandle, idleStateType));

    stateManager.addState(
        make_unique<CurrentWeatherState>(language, stateManager, desireSet, nodeHandle, afterTaskDelayStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        afterTaskDelayStateType,
        songPaths));

    stateManager.addState(make_unique<AfterTaskDelayState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        idleStateType,
        afterTaskDelayDuration));

    stateManager.switchTo<SmartIdleState>();

    ros::spin();
}

int main(int argc, char** argv)
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

    double personDistanceThreshold = -1.0;
    if (!privateNodeHandle.getParam("person_distance_threshold", personDistanceThreshold) ||
        personDistanceThreshold < 0.0)
    {
        ROS_ERROR("The parameter person_distance_threshold must be set and greater than 0.");
        return -1;
    }

    std::string personDistanceFrame;
    if (!privateNodeHandle.getParam("person_distance_frame", personDistanceFrame) || personDistanceFrame == "")
    {
        ROS_ERROR("The parameter person_distance_frame must be set and not empty.");
        return -1;
    }

    double noseConfidenceThreshold = -1.0;
    if (!privateNodeHandle.getParam("nose_confidence_threshold", noseConfidenceThreshold) ||
        noseConfidenceThreshold < 0.0)
    {
        ROS_ERROR("The parameter nose_confidence_threshold must be set and not empty.");
        return -1;
    }

    int videoAnalysisMessageCountThreshold = -1;
    if (!privateNodeHandle.getParam("video_analysis_message_count_threshold", videoAnalysisMessageCountThreshold) ||
        videoAnalysisMessageCountThreshold < 1)
    {
        ROS_ERROR("The parameter video_analysis_message_count_threshold must be set and greater than 0.");
        return -1;
    }

    int videoAnalysisMessageCountTolerance = -1;
    if (!privateNodeHandle.getParam("video_analysis_message_count_tolerance", videoAnalysisMessageCountTolerance) ||
        videoAnalysisMessageCountTolerance < 0)
    {
        ROS_ERROR("The parameter video_analysis_message_count_tolerance must be set and greater than or equal to 0.");
        return -1;
    }

    vector<string> songNames;
    if (!privateNodeHandle.getParam("song_names", songNames) || songNames.size() == 0)
    {
        ROS_ERROR("At least one valid name must be set for the songs.");
        return -1;
    }

    vector<string> songPaths;
    if (!privateNodeHandle.getParam("song_paths", songPaths) || songPaths.size() == 0)
    {
        ROS_ERROR("At least one valid path must be set for the songs.");
        return -1;
    }

    if (songNames.size() != songPaths.size())
    {
        ROS_ERROR("The size of song_names and song_paths must be the same.");
        return -1;
    }

    double afterTaskDelayDurationS;
    privateNodeHandle.param("after_task_delay_duration_s", afterTaskDelayDurationS, 0.0);
    ros::Duration afterTaskDelayDuration(afterTaskDelayDurationS);

    startNode(
        language,
        nodeHandle,
        personDistanceThreshold,
        personDistanceFrame,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance,
        songNames,
        songPaths,
        afterTaskDelayDuration);

    return 0;
}
