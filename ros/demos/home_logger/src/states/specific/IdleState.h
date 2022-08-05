#ifndef HOME_LOGGER_STATES_SPECIFIC_IDLE_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_IDLE_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/DateTime.h>

#include <chrono>

class IdleState : public SoundFaceFollowingState
{
    Time m_sleepTime;
    Time m_wakeUpTime;

    tl::optional<uint64_t> m_faceAnimationDesireId;

    std::chrono::time_point<std::chrono::system_clock> m_lastGreetingTime;

    bool m_chargeNeeded;
    std::chrono::time_point<std::chrono::system_clock> m_lastChargingMessageTime;

public:
    IdleState(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle, Time sleepTime, Time wakeUpTime);
    ~IdleState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(IdleState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg) override;
    void onRobotNameDetected() override;
    void onBaseStatusChanged(
        float stateOfCharge,
        float voltage,
        float current,
        bool isPsuConnected,
        bool isBatteryCharging) override;
    void onEveryMinuteTimeout() override;
};

#endif
