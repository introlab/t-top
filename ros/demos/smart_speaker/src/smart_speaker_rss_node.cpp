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
constexpr const char* NODE_NAME = "smart_speaker_rss_node";

void startNode(
    Language language,
    rclcpp::Node::SharedPtr node,
    const string& englishStoryPath,
    const string& frenchStoryPath,
    const string& songPath,
    bool useAfterTaskDelayDurationTopic,
    const chrono::milliseconds& afterTaskDelayDurationMs)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(node, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createSlowVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, node));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, node));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    StateManager stateManager(node);
    type_index idleStateType(typeid(RssIdleState));
    type_index afterTaskDelayStateType(typeid(AfterTaskDelayState));

    stateManager.addState(make_unique<RssIdleState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<RssWaitPersonIdentificationState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<RssAskTaskState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<RssWaitAnswerState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<RssValidTaskState>(language, stateManager, desireSet, node));
    stateManager.addState(make_unique<InvalidTaskState>(language, stateManager, desireSet, node, idleStateType));

    stateManager.addState(
        make_unique<CurrentWeatherState>(language, stateManager, desireSet, node, afterTaskDelayStateType));
    stateManager.addState(
        make_unique<WeatherForecastState>(language, stateManager, desireSet, node, afterTaskDelayStateType));
    stateManager.addState(
        make_unique<RssStoryState>(language, stateManager, desireSet, node, englishStoryPath, frenchStoryPath));
    stateManager.addState(
        make_unique<DanceState>(language, stateManager, desireSet, node, afterTaskDelayStateType));
    stateManager.addState(make_unique<DancePlayedSongState>(
        language,
        stateManager,
        desireSet,
        node,
        afterTaskDelayStateType,
        vector<string>{songPath}));

    stateManager.addState(make_unique<AfterTaskDelayState>(
        language,
        stateManager,
        desireSet,
        node,
        idleStateType,
        useAfterTaskDelayDurationTopic,
        afterTaskDelayDurationMs));

    stateManager.switchTo<RssIdleState>();

    rclcpp::spin(node);
}

int startNode()
{
    auto node = rclcpp::Node::make_shared(NODE_NAME);

    string languageString = node->declare_parameter("language", "");
    Language language;
    if (!languageFromString(languageString, language))
    {
        RCLCPP_ERROR(node->get_logger(), "Language must be English (language=en) or French (language=fr).");
        return -1;
    }

    string englishStoryPath = node->declare_parameter("story_path_en", "");
    if (englishStoryPath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "A valid path must be set for the English story.");
        return -1;
    }

    string frenchStoryPath = node->declare_parameter("story_path_fr", "");
    if (frenchStoryPath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "A valid path must be set for the French story.");
        return -1;
    }

    string songPath = node->declare_parameter("song_path", "");
    if (songPath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "A valid path must be set for the song.");
        return -1;
    }

    bool useAfterTaskDelayDurationTopic = node->declare_parameter("use_after_task_delay_duration_topic", false);
    double afterTaskDelayDurationS = node->declare_parameter("after_task_delay_duration_s", 0.0);
    chrono::milliseconds afterTaskDelayDurationMs(static_cast<int>(afterTaskDelayDurationS * 1000));

    startNode(
        language,
        node,
        englishStoryPath,
        frenchStoryPath,
        songPath,
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
