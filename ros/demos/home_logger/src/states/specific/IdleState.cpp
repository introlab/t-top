#include "IdleState.h"
#include "SleepState.h"
#include "WaitCommandState.h"
#include "AlarmState.h"
#include "TellReminderState.h"
#include "../StateManager.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringResources.h>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

constexpr float LOW_STATE_OF_CHARGE = 25;
constexpr chrono::minutes BATTERY_LOW_MESSAGE_INTERVAL(5);
constexpr chrono::hours GREATING_INTERVAL(4);

IdleState::IdleState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    AlarmManager& alarmManager,
    ReminderManager& reminderManager,
    Time sleepTime,
    Time wakeUpTime,
    float faceDescriptorThreshold)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_alarmManager(alarmManager),
      m_reminderManager(reminderManager),
      m_sleepTime(sleepTime),
      m_wakeUpTime(wakeUpTime),
      m_faceDescriptorThreshold(faceDescriptorThreshold),
      m_chargeNeeded(false)
{
}

IdleState::~IdleState() {}

void IdleState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("blink");
    m_robotNameDetectorDesireId = m_desireSet->addDesire<RobotNameDetectorDesire>();

    if (!m_desireSet->containsAnyDesiresOfType<FastVideoAnalyzer3dDesire>())
    {
        m_videoAnalyzer3dDesireId = m_desireSet->addDesire<FastVideoAnalyzer3dDesire>();
    }

    m_todayReminders = m_reminderManager.listReminders(Date::now());
}

void IdleState::onDisabling()
{
    SoundFaceFollowingState::onDisabling();

    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = nullopt;
    }
    if (m_robotNameDetectorDesireId.has_value())
    {
        m_desireSet->removeDesire(m_robotNameDetectorDesireId.value());
        m_robotNameDetectorDesireId = nullopt;
    }
    if (m_videoAnalyzer3dDesireId.has_value())
    {
        m_desireSet->removeDesire(m_videoAnalyzer3dDesireId.value());
        m_videoAnalyzer3dDesireId = nullopt;
    }
}

void IdleState::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    SoundFaceFollowingState::onVideoAnalysisReceived(msg);

    auto reminder = findReminder(msg);
    if (reminder.has_value())
    {
        m_stateManager.switchTo<TellReminderState>(TellReminderStateParameter(reminder.value()));
        return;
    }

    bool atLeastOnePerson = containsAtLeastOnePerson(msg);
    auto now = chrono::system_clock::now();
    if (atLeastOnePerson && m_chargeNeeded && (now - m_lastChargingMessageTime) >= BATTERY_LOW_MESSAGE_INTERVAL)
    {
        m_lastChargingMessageTime = now;
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.idle_state.low_battery"),
            "",  // No gesture
            "fear",
            StateType::get<IdleState>()));
        return;
    }

    if (atLeastOnePerson && (now - m_lastGreetingTime) >= GREATING_INTERVAL)
    {
        m_lastGreetingTime = now;
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.idle_state.hi") + " " +
                StringResources::getValue("dialogs.idle_state.ask_command"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandState>()));
        return;
    }
}

void IdleState::onRobotNameDetected()
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        StringResources::getValue("dialogs.idle_state.ask_command"),
        "",  // No gesture
        "blink",
        StateType::get<WaitCommandState>()));
}

void IdleState::onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg)
{
    m_chargeNeeded = msg->state_of_charge <= LOW_STATE_OF_CHARGE && !msg->is_psu_connected;
}

void IdleState::onEveryMinuteTimeout()
{
    Time now = Time::now();
    if (now.between(m_sleepTime, m_wakeUpTime))
    {
        m_stateManager.switchTo<SleepState>();
        return;
    }

    auto alarms = m_alarmManager.listDueAlarms(DateTime::now());
    if (!alarms.empty())
    {
        switchToAlarmState(move(alarms));
        return;
    }
}

void IdleState::onEveryTenMinutesTimeout()
{
    m_reminderManager.removeRemindersOlderThan(DateTime::now());
    m_todayReminders = m_reminderManager.listReminders(Date::now());
}

optional<Reminder> IdleState::findReminder(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    for (auto& object : msg->objects)
    {
        if (object.object_class != "person" || object.face_descriptor.empty())
        {
            continue;
        }

        for (auto& reminder : m_todayReminders)
        {
            if (reminder.faceDescriptor().distance(object.face_descriptor) <= m_faceDescriptorThreshold)
            {
                return reminder;
            }
        }
    }

    return nullopt;
}

void IdleState::switchToAlarmState(vector<unique_ptr<Alarm>> alarms)
{
    vector<int> alarmIds;
    for (const auto& alarm : alarms)
    {
        if (alarm->id().has_value())
        {
            alarmIds.emplace_back(alarm->id().value());
        }
    }
    m_stateManager.switchTo<AlarmState>(AlarmStateParameter(alarmIds));
}
