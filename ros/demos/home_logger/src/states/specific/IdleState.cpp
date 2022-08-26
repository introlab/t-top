#include "IdleState.h"
#include "SleepState.h"
#include "WaitCommandState.h"
#include "TellReminderState.h"
#include "../StateManager.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringRessources.h>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

constexpr float LOW_STATE_OF_CHARGE = 25;
constexpr chrono::minutes BATTERY_LOW_MESSAGE_INTERVAL(5);

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
    m_todayReminders = m_reminderManager.listReminders(Date::now());
}

void IdleState::onDisabling()
{
    SoundFaceFollowingState::onDisabling();

    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = tl::nullopt;
    }
}

void IdleState::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    SoundFaceFollowingState::onVideoAnalysisReceived(msg);

    auto now = chrono::system_clock::now();
    if (containsAtLeastOnePerson(msg) && m_chargeNeeded && (now - m_lastChargingMessageTime) >= BATTERY_LOW_MESSAGE_INTERVAL)
    {
        m_lastChargingMessageTime = now;
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringRessources::getValue("dialogs.idle_state.low_battery"),
            "",  // No gesture
            "fear",
            StateType::get<IdleState>()));
    }

    // TODO greatings

    auto reminder = findReminder(msg);
    if (reminder.has_value())
    {
        m_stateManager.switchTo<TellReminderState>(TellReminderStateParameter(reminder.value()));
    }
}

void IdleState::onRobotNameDetected()
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        StringRessources::getValue("dialogs.idle_state.ask_command"),
        "",  // No gesture
        "blink",
        StateType::get<WaitCommandState>()));
}

void IdleState::onBaseStatusChanged(
    float stateOfCharge,
    float voltage,
    float current,
    bool isPsuConnected,
    bool isBatteryCharging)
{
    m_chargeNeeded = stateOfCharge <= LOW_STATE_OF_CHARGE && !isPsuConnected;
}

void IdleState::onEveryMinuteTimeout()
{
    Time now = Time::now();
    if (now.between(m_sleepTime, m_wakeUpTime))
    {
        m_stateManager.switchTo<SleepState>();
        return;
    }

    // TODO check alarms
}

tl::optional<Reminder> IdleState::findReminder(const video_analyzer::VideoAnalysis::ConstPtr& msg)
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

    return tl::nullopt;
}
