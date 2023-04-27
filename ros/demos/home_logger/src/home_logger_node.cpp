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
constexpr double STARTUP_DELAY_S = 30.0;

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
    ros::NodeHandle& nodeHandle,
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
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

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

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, nodeHandle));

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosStrategyStateLogger>(nodeHandle);
    HbbaLite hbba(desireSet, move(strategies), {{"motor", 1}, {"sound", 1}}, move(solver), move(strategyStateLogger));

    VolumeManager volumeManager(nodeHandle);
    SQLite::Database database(databasePath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    AlarmManager alarmManager(database);
    ReminderManager reminderManager(database);

    StateManager stateManager(desireSet, nodeHandle);
    stateManager.addState(make_unique<TalkState>(stateManager, desireSet, nodeHandle));

    stateManager.addState(make_unique<IdleState>(
        stateManager,
        desireSet,
        nodeHandle,
        alarmManager,
        reminderManager,
        sleepTime,
        wakeUpTime,
        faceDescriptorThreshold));
    stateManager.addState(make_unique<SleepState>(stateManager, desireSet, nodeHandle, sleepTime, wakeUpTime));
    stateManager.addState(make_unique<WaitCommandState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<ExecuteCommandState>(
        stateManager,
        desireSet,
        nodeHandle,
        volumeManager,
        alarmManager,
        reminderManager));
    stateManager.addState(make_unique<WaitCommandParameterState>(stateManager, desireSet, nodeHandle));
    stateManager.addState(make_unique<WaitFaceDescriptorCommandParameterState>(
        stateManager,
        desireSet,
        nodeHandle,
        noseConfidenceThreshold));

    stateManager.addState(make_unique<AlarmState>(stateManager, desireSet, nodeHandle, alarmManager, alarmPath));
    stateManager.addState(make_unique<TellReminderState>(stateManager, desireSet, nodeHandle, reminderManager));


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

    ros::spin();
}

int startNode(int argc, char** argv)
{
    ros::init(argc, argv, "home_logger_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    string languageString;
    Language language;
    privateNodeHandle.param<string>("language", languageString, "");
    if (!languageFromString(languageString, language))
    {
        ROS_ERROR("Language must be English (language=en) or French (language=fr).");
        return -1;
    }

    string englishStringResourcePath;
    if (!privateNodeHandle.getParam("english_string_resource_path", englishStringResourcePath))
    {
        ROS_ERROR("The parameter english_string_resource_path must be set.");
        return -1;
    }
    string frenchStringResourcesPath;
    if (!privateNodeHandle.getParam("french_string_resources_path", frenchStringResourcesPath))
    {
        ROS_ERROR("The parameter french_string_resources_path must be set.");
        return -1;
    }

    string databasePath;
    if (!privateNodeHandle.getParam("database_path", databasePath))
    {
        ROS_ERROR("The parameter database_path is required.");
        return -1;
    }

    int sleepTimeHour;
    int sleepTimeMinute;
    int wakeUpTimeHour;
    int wakeUpTimeMinute;
    if (!privateNodeHandle.getParam("sleep_time_hour", sleepTimeHour))
    {
        ROS_ERROR("The parameter sleep_time_hour must be set.");
        return -1;
    }
    if (!privateNodeHandle.getParam("sleep_time_minute", sleepTimeMinute))
    {
        ROS_ERROR("The parameter sleep_time_minute must be set.");
        return -1;
    }
    if (!privateNodeHandle.getParam("wake_up_time_hour", wakeUpTimeHour))
    {
        ROS_ERROR("The parameter wake_up_time_hour must be set.");
        return -1;
    }
    if (!privateNodeHandle.getParam("wake_up_time_minute", wakeUpTimeMinute))
    {
        ROS_ERROR("The parameter wake_up_time_minute must be set.");
        return -1;
    }

    string alarmPath;
    if (!privateNodeHandle.getParam("alarm_path", alarmPath))
    {
        ROS_ERROR("The parameter alarm_path must be set.");
        return -1;
    }

    float faceDescriptorThreshold;
    if (!privateNodeHandle.getParam("face_descriptor_threshold", faceDescriptorThreshold))
    {
        ROS_ERROR("The parameter face_descriptor_threshold must be set.");
        return -1;
    }

    float noseConfidenceThreshold;
    if (!privateNodeHandle.getParam("nose_confidence_threshold", noseConfidenceThreshold))
    {
        ROS_ERROR("The parameter nose_confidence_threshold must be set.");
        return -1;
    }


    bool camera2dWideEnabled;
    if (!privateNodeHandle.getParam("camera_2d_wide_enabled", camera2dWideEnabled))
    {
        ROS_ERROR("The parameter camera_2d_wide_enabled must be set.");
        return -1;
    }

    bool recordSession;
    if (!privateNodeHandle.getParam("record_session", recordSession))
    {
        ROS_ERROR("The parameter record_session must be set.");
        return -1;
    }

    bool logPerceptions;
    if (!privateNodeHandle.getParam("log_perceptions", logPerceptions))
    {
        ROS_ERROR("The parameter log_perceptions must be set.");
        return -1;
    }

    ROS_INFO("Waiting nodes.");
    ros::Duration(STARTUP_DELAY_S).sleep();

    startNode(
        nodeHandle,
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
    try
    {
        return startNode(argc, argv);
    }
    catch (const exception& e)
    {
        ROS_ERROR_STREAM("Home logger crashed (" << e.what() << ")");
        return -1;
    }
}
