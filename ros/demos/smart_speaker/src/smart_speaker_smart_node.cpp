#include "StringUtils.h"

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

#include <rclcpp/rclcpp.hpp>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "smart_speaker_smart_node";

void logSongs(
    rclcpp::Node::SharedPtr& node,
    const vector<string>& songNames,
    const vector<vector<string>>& songKeywords,
    const vector<string>& songPaths)
{
    for (size_t i = 0; i < songNames.size(); i++)
    {
        RCLCPP_INFO_STREAM(node->get_logger(), "Song " << (i + 1));
        RCLCPP_INFO_STREAM(node->get_logger(), "\tname=" << songNames[i]);
        RCLCPP_INFO_STREAM(node->get_logger(), "\tkeywords=" << mergeStrings(songKeywords[i], ","));
        RCLCPP_INFO_STREAM(node->get_logger(), "\tpath=" << songPaths[i]);
    }
}

void startNode(
    bool recordSession,
    Language language,
    rclcpp::Node::SharedPtr node,
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
    const chrono::milliseconds& afterTaskDelayDurationMs)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(node, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));

    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, node));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, node));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    StateManager stateManager(node);
    type_index askOtherTaskStateType(typeid(SmartAskOtherTaskState));

    stateManager.addState(make_unique<SmartIdleState>(
        language,
        stateManager,
        desireSet,
        node,
        personDistanceThreshold,
        personDistanceFrameId,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance));
    stateManager.addState(make_unique<SmartAskTaskState>(language, stateManager, desireSet, node, songNames));
    stateManager.addState(make_unique<SmartWaitAnswerState>(language, stateManager, desireSet, node, songKeywords));
    stateManager.addState(make_unique<SmartValidTaskState>(language, stateManager, desireSet, node));
    stateManager.addState(
        make_unique<InvalidTaskState>(language, stateManager, desireSet, node, askOtherTaskStateType));

    stateManager.addState(
        make_unique<CurrentWeatherState>(language, stateManager, desireSet, node, askOtherTaskStateType));
    stateManager.addState(
        make_unique<DancePlayedSongState>(language, stateManager, desireSet, node, askOtherTaskStateType, songPaths));

    stateManager.addState(
        make_unique<SmartAskOtherTaskState>(language, stateManager, desireSet, node, singleTaskPerPerson));
    stateManager.addState(make_unique<SmartThankYouState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<AfterTaskDelayState>(
        language,
        stateManager,
        desireSet,
        node,
        typeid(SmartIdleState),
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDurationMs));

    stateManager.switchTo<SmartIdleState>();

    if (recordSession)
    {
        desireSet->addDesire(make_unique<Camera3dRecordingDesire>());
    }

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
}

int startNode()
{
    auto node = rclcpp::Node::make_shared(NODE_NAME);

    bool recordSession = node->declare_parameter("record_session", false);

    string languageString = node->declare_parameter("language", "");
    Language language;
    if (!languageFromString(languageString, language))
    {
        RCLCPP_ERROR(node->get_logger(), "Language must be English (language=en) or French (language=fr).");
        return -1;
    }

    double personDistanceThreshold = node->declare_parameter("person_distance_threshold", -1.0);
    if (personDistanceThreshold < 0.0)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter person_distance_threshold must be set and greater than 0.");
        return -1;
    }

    std::string personDistanceFrameId = node->declare_parameter("person_distance_frame_id", "");
    if (personDistanceFrameId == "")
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter person_distance_frame_id must be set and not empty.");
        return -1;
    }

    double noseConfidenceThreshold = node->declare_parameter("nose_confidence_threshold", -1.0);
    if (noseConfidenceThreshold < 0.0)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter nose_confidence_threshold must be set and not empty.");
        return -1;
    }

    int videoAnalysisMessageCountThreshold = node->declare_parameter("video_analysis_message_count_threshold", -1);
    if (videoAnalysisMessageCountThreshold < 1)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "The parameter video_analysis_message_count_threshold must be set and greater than 0.");
        return -1;
    }

    int videoAnalysisMessageCountTolerance = node->declare_parameter("video_analysis_message_count_tolerance", -1);
    if (videoAnalysisMessageCountTolerance < 0)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "The parameter video_analysis_message_count_tolerance must be set and greater than or equal to 0.");
        return -1;
    }

    bool singleTaskPerPerson = node->declare_parameter("single_task_per_person", false);
    bool useAfterTaskDelayDurationTopic = node->declare_parameter("use_after_task_delay_duration_topic", false);

    double afterTaskDelayDurationS = node->declare_parameter("after_task_delay_duration_s", 0.0);
    chrono::milliseconds afterTaskDelayDurationMs(static_cast<int>(afterTaskDelayDurationS * 1000));

    vector<string> songNames = node->declare_parameter("song_names", vector<string>{});
    vector<string> songKeywords = node->declare_parameter("song_keywords", vector<string>{});
    vector<string> songPaths = node->declare_parameter("song_paths", vector<string>{});
    if (songNames.size() != songKeywords.size() || songNames.size() != songPaths.size() || songNames.size() < 1)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "The parameters song_names, song_keywords and song_paths must have the same size and contain at least one "
            "item.");
        return -1;
    }

    vector<vector<string>> splittedSongKeywords;
    std::transform(
        songKeywords.begin(),
        songKeywords.end(),
        std::back_inserter(splittedSongKeywords),
        [](auto& x) { return splitStrings(x, ";"); });


    logSongs(node, songNames, splittedSongKeywords, songPaths);
    startNode(
        recordSession,
        language,
        node,
        personDistanceThreshold,
        personDistanceFrameId,
        noseConfidenceThreshold,
        videoAnalysisMessageCountThreshold,
        videoAnalysisMessageCountTolerance,
        songNames,
        splittedSongKeywords,
        songPaths,
        singleTaskPerPerson,
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDurationMs);

    return 0;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        return startNode();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger(NODE_NAME), "Smart speaker crashed (" << e.what() << ")");
        return -1;
    }

    rclcpp::shutdown();
}
