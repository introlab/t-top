#include "states/StateManager.h"

#include "states/smart/SmartIdleState.h"
#include "states/smart/SmartAskTaskState.h"
#include "states/smart/SmartWaitAnswerState.h"
#include "states/smart/SmartValidTaskState.h"
#include "states/common/InvalidTaskState.h"

#include "states/task/CurrentWeatherState.h"
#include "states/task/DancePlayedSongState.h"

#include "states/smart/SmartAskOtherTaskState.h"
#include "states/smart/SmartThankYouState.h"
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
    bool recordSession,
    Language language,
    ros::NodeHandle& nodeHandle,
    double personDistanceThreshold,
    const std::string& personDistanceFrameId,
    double noseConfidenceThreshold,
    size_t videoAnalysisMessageCountThreshold,
    size_t videoAnalysisMessageCountTolerance,
    const vector<string>& songNames,
    const vector<vector<string>>& songKeywords,
    const vector<string>& songPaths,
    bool singleTaskPerPerson,
    bool useAfterTaskDelayDurationTopic,
    const ros::Duration& afterTaskDelayDuration)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));

    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, nodeHandle));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(nodeHandle);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    StateManager stateManager;
    type_index askOtherTaskStateType(typeid(SmartAskOtherTaskState));

    stateManager.addState(make_unique<SmartIdleState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        personDistanceThreshold,
        personDistanceFrameId,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance));
    stateManager.addState(make_unique<SmartAskTaskState>(language, stateManager, desireSet, nodeHandle, songNames));
    stateManager.addState(
        make_unique<SmartWaitAnswerState>(language, stateManager, desireSet, nodeHandle, songKeywords));
    stateManager.addState(make_unique<SmartValidTaskState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(
        make_unique<InvalidTaskState>(language, stateManager, desireSet, nodeHandle, askOtherTaskStateType));

    stateManager.addState(
        make_unique<CurrentWeatherState>(language, stateManager, desireSet, nodeHandle, askOtherTaskStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        askOtherTaskStateType,
        songPaths));

    stateManager.addState(
        make_unique<SmartAskOtherTaskState>(language, stateManager, desireSet, nodeHandle, singleTaskPerPerson));
    stateManager.addState(make_unique<SmartThankYouState>(language, stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<AfterTaskDelayState>(
        language,
        stateManager,
        desireSet,
        nodeHandle,
        typeid(SmartIdleState),
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDuration));

    stateManager.switchTo<SmartIdleState>();

    if (recordSession)
    {
        desireSet->addDesire(make_unique<Camera3dRecordingDesire>());
    }

    ros::spin();
}

bool getSongStrings(ros::NodeHandle& privateNodeHandle, vector<string>& values, const std::string& key)
{
    string value;

    XmlRpc::XmlRpcValue songs;
    privateNodeHandle.getParam("songs", songs);
    if (songs.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        ROS_ERROR("Invalid songs format");
        return false;
    }

    for (size_t i = 0; i < songs.size(); i++)
    {
        if (!songs[i].hasMember(key) || (value = static_cast<string>(songs[i][key])).empty())
        {
            ROS_ERROR_STREAM("Invalid songs[" << i << "] " << key);
            return false;
        }

        values.emplace_back(move(value));
    }

    return values.size() > 0;
}

bool getSongVectors(ros::NodeHandle& privateNodeHandle, vector<vector<string>>& values, const std::string& key)
{
    vector<string> value;

    XmlRpc::XmlRpcValue songs;
    privateNodeHandle.getParam("songs", songs);
    if (songs.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        ROS_ERROR("Invalid songs format");
        return false;
    }

    for (size_t i = 0; i < songs.size(); i++)
    {
        if (!songs[i].hasMember(key) || songs[i][key].getType() != XmlRpc::XmlRpcValue::TypeArray)
        {
            ROS_ERROR_STREAM("Invalid songs[" << i << "] " << key);
            return false;
        }

        for (size_t j = 0; j < songs[i][key].size(); j++)
        {
            value.emplace_back(songs[i][key][j]);
        }

        values.emplace_back(move(value));
    }

    return values.size() > 0;
}

int startNode(int argc, char** argv)
{
    ros::init(argc, argv, "smart_speaker_smart_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    bool recordSession;
    if (!privateNodeHandle.getParam("record_session", recordSession))
    {
        ROS_ERROR("The parameter record_session must be set.");
        return -1;
    }

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

    std::string personDistanceFrameId;
    if (!privateNodeHandle.getParam("person_distance_frame_id", personDistanceFrameId) || personDistanceFrameId == "")
    {
        ROS_ERROR("The parameter person_distance_frame_id must be set and not empty.");
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

    bool singleTaskPerPerson = false;
    privateNodeHandle.param("single_task_per_person", singleTaskPerPerson, false);

    bool useAfterTaskDelayDurationTopic = false;
    privateNodeHandle.param("use_after_task_delay_duration_topic", useAfterTaskDelayDurationTopic, false);

    double afterTaskDelayDurationS;
    privateNodeHandle.param("after_task_delay_duration_s", afterTaskDelayDurationS, 0.0);
    ros::Duration afterTaskDelayDuration(afterTaskDelayDurationS);

    vector<string> songNames;
    vector<vector<string>> songKeywords;
    vector<string> songPaths;
    if (!getSongStrings(privateNodeHandle, songNames, "name") ||
        !getSongVectors(privateNodeHandle, songKeywords, "keywords") ||
        !getSongStrings(privateNodeHandle, songPaths, "path"))
    {
        return -1;
    }

    startNode(
        recordSession,
        language,
        nodeHandle,
        personDistanceThreshold,
        personDistanceFrameId,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance,
        songNames,
        songKeywords,
        songPaths,
        singleTaskPerPerson,
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
