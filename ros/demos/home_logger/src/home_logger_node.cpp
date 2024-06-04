#include "states/StateManager.h"

#include "managers/VolumeManager.h"

#include "states/common/TalkState.h"
#include "states/specific/IdleState.h"
#include "states/specific/SleepState.h"
#include "states/specific/WaitCommandState.h"
#include "states/specific/ExecuteCommandState.h"
#include "states/specific/WaitCommandParameterState.h"
#include "states/specific/WaitFaceDescriptorCommandParameterState.h"
#include "states/specific/AlarmState.h"
#include "states/specific/TellReminderState.h"

#include <home_logger_common/language/Language.h>
#include <home_logger_common/language/StringResources.h>
#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/managers/AlarmManager.h>

#include <rclcpp/rclcpp.hpp>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>
#include <thread>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr int STARTUP_DELAY_S = 30;
constexpr const char* NODE_NAME = "home_logger_node";

void loadResources(Language language, const string& englishStringResourcePath, const string& frenchStringResourcesPath)
{
    if (language == Language::ENGLISH)
    {
        StringResources::loadFromFile(englishStringResourcePath, Language::ENGLISH);
    }
    else if (language == Language::FRENCH)
    {
        StringResources::loadFromFile(frenchStringResourcesPath, Language::FRENCH);
    }
    else
    {
        throw runtime_error("Invalid language");
    }
}

void startNode(
    rclcpp::Node::SharedPtr node,
    Language language,
    const string& englishStringResourcePath,
    const string& frenchStringResourcesPath,
    const string& databasePath,
    bool camera2dWideEnabled,
    bool recordSession,
    bool logPerceptions,
    Time sleepTime,
    Time wakeUpTime,
    const string& alarmPath,
    float faceDescriptorThreshold,
    float noseConfidenceThreshold)
{
    loadResources(language, englishStringResourcePath, frenchStringResourcesPath);
    Formatter::initialize(language);

    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(node, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    if (recordSession)
    {
        strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));
    }
    if (recordSession && camera2dWideEnabled)
    {
        strategies.emplace_back(createCamera2dWideRecordingStrategy(filterPool));
    }

    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, node));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, node));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    VolumeManager volumeManager(node);
    SQLite::Database database(databasePath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    AlarmManager alarmManager(database);
    ReminderManager reminderManager(database);

    StateManager stateManager(desireSet, node);
    stateManager.addState(make_unique<TalkState>(stateManager, desireSet, node));

    stateManager.addState(make_unique<IdleState>(
        stateManager,
        desireSet,
        node,
        alarmManager,
        reminderManager,
        sleepTime,
        wakeUpTime,
        faceDescriptorThreshold));
    stateManager.addState(make_unique<SleepState>(stateManager, desireSet, node, sleepTime, wakeUpTime));
    stateManager.addState(make_unique<WaitCommandState>(stateManager, desireSet, node));
    stateManager.addState(
        make_unique<ExecuteCommandState>(stateManager, desireSet, node, volumeManager, alarmManager, reminderManager));
    stateManager.addState(make_unique<WaitCommandParameterState>(stateManager, desireSet, node));
    stateManager.addState(
        make_unique<WaitFaceDescriptorCommandParameterState>(stateManager, desireSet, node, noseConfidenceThreshold));

    stateManager.addState(make_unique<AlarmState>(stateManager, desireSet, node, alarmManager, alarmPath));
    stateManager.addState(make_unique<TellReminderState>(stateManager, desireSet, node, reminderManager));


    if (recordSession)
    {
        desireSet->addDesire<Camera3dRecordingDesire>();
    }
    if (recordSession && camera2dWideEnabled)
    {
        desireSet->addDesire<Camera2dWideRecordingDesire>();
    }

    if (logPerceptions)
    {
        desireSet->addDesire<AudioAnalyzerDesire>();
        desireSet->addDesire<FastVideoAnalyzer3dDesire>();
    }

    stateManager.switchTo<IdleState>();

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

    string englishStringResourcePath = node->declare_parameter("english_string_resource_path", "");
    if (englishStringResourcePath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter english_string_resource_path must be set.");
        return -1;
    }
    string frenchStringResourcesPath = node->declare_parameter("french_string_resources_path", "");
    if (frenchStringResourcesPath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter french_string_resources_path must be set.");
        return -1;
    }

    string databasePath = node->declare_parameter("database_path", "");
    if (databasePath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter database_path is required.");
        return -1;
    }

    int sleepTimeHour = node->declare_parameter("sleep_time_hour", -1);
    int sleepTimeMinute = node->declare_parameter("sleep_time_minute", -1);
    int wakeUpTimeHour = node->declare_parameter("wake_up_time_hour", -1);
    int wakeUpTimeMinute = node->declare_parameter("wake_up_time_minute", -1);
    if (sleepTimeHour == -1)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter sleep_time_hour must be set.");
        return -1;
    }
    if (sleepTimeMinute == -1)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter sleep_time_minute must be set.");
        return -1;
    }
    if (wakeUpTimeHour == -1)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter wake_up_time_hour must be set.");
        return -1;
    }
    if (wakeUpTimeMinute == -1)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter wake_up_time_minute must be set.");
        return -1;
    }

    string alarmPath = node->declare_parameter("alarm_path", "");
    if (alarmPath == "")
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter alarm_path must be set.");
        return -1;
    }

    float faceDescriptorThreshold = node->declare_parameter("face_descriptor_threshold", -1.f);
    if (faceDescriptorThreshold == -1.f)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter face_descriptor_threshold must be set.");
        return -1;
    }

    float noseConfidenceThreshold = node->declare_parameter("nose_confidence_threshold", -1.f);
    if (noseConfidenceThreshold == -1.f)
    {
        RCLCPP_ERROR(node->get_logger(), "The parameter nose_confidence_threshold must be set.");
        return -1;
    }


    bool camera2dWideEnabled = node->declare_parameter("camera_2d_wide_enabled", false);
    bool recordSession = node->declare_parameter("record_session", false);
    bool logPerceptions = node->declare_parameter("log_perceptions", false);

    RCLCPP_INFO(node->get_logger(), "Waiting nodes.");
    this_thread::sleep_for(chrono::seconds(STARTUP_DELAY_S));

    startNode(
        node,
        language,
        englishStringResourcePath,
        frenchStringResourcesPath,
        databasePath,
        camera2dWideEnabled,
        recordSession,
        logPerceptions,
        Time(sleepTimeHour, sleepTimeMinute),
        Time(wakeUpTimeHour, wakeUpTimeMinute),
        alarmPath,
        faceDescriptorThreshold,
        noseConfidenceThreshold);

    return 0;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        return startNode();
    }
    catch (const exception& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger(NODE_NAME), "Home logger crashed (" << e.what() << ")");
        return -1;
    }

    rclcpp::shutdown();
}
