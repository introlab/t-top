#ifndef HOME_LOGGER_STATES_SPECIFIC_SLEEP_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_SLEEP_STATE_H

#include "../State.h"

#include <home_logger_common/DateTime.h>

#include <optional>
#include <chrono>

class SleepState : public State
{
    Time m_sleepTime;
    Time m_wakeUpTime;

    bool m_wasForced;

    bool m_hadCamera3dRecordingDesire;
    bool m_hadCamera2dWideRecordingDesire;
    bool m_hadAudioAnalyzerDesire;
    bool m_hadFastVideoAnalyzer3dDesire;

    std::optional<uint64_t> m_faceAnimationDesireId;

public:
    SleepState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        Time sleepTime,
        Time wakeUpTime);
    ~SleepState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(SleepState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onEveryMinuteTimeout() override;
};

#endif
