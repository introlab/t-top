#ifndef HOME_LOGGER_STATES_SPECIFIC_IDLE_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_IDLE_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/DateTime.h>
#include <home_logger_common/managers/AlarmManager.h>
#include <home_logger_common/managers/ReminderManager.h>

#include <chrono>

class IdleState : public SoundFaceFollowingState
{
    AlarmManager& m_alarmManager;
    ReminderManager& m_reminderManager;

    Time m_sleepTime;
    Time m_wakeUpTime;
    float m_faceDescriptorThreshold;

    std::optional<uint64_t> m_faceAnimationDesireId;
    std::optional<uint64_t> m_robotNameDetectorDesireId;
    std::optional<uint64_t> m_videoAnalyzer3dDesireId;

    bool m_chargeNeeded;
    std::chrono::time_point<std::chrono::system_clock> m_lastChargingMessageTime;
    std::chrono::time_point<std::chrono::system_clock> m_lastGreetingTime;

    std::vector<Reminder> m_todayReminders;

public:
    IdleState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        AlarmManager& alarmManager,
        ReminderManager& reminderManager,
        Time sleepTime,
        Time wakeUpTime,
        float faceDescriptorThreshold);
    ~IdleState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(IdleState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg) override;
    void onRobotNameDetected() override;
    void onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg) override;
    void onEveryMinuteTimeout() override;
    void onEveryTenMinutesTimeout() override;

private:
    std::optional<Reminder> findReminder(const video_analyzer::VideoAnalysis::ConstPtr& msg);
    void switchToAlarmState(std::vector<std::unique_ptr<Alarm>> alarms);
};

#endif
