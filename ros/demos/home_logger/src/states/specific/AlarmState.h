#ifndef HOME_LOGGER_STATES_SPECIFIC_ALARM_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_ALARM_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/managers/AlarmManager.h>

class AlarmStateParameter : public StateParameter
{
public:
    std::vector<int> alarmIds;

    AlarmStateParameter();
    explicit AlarmStateParameter(std::vector<int> alarmIds);
    ~AlarmStateParameter() override;

    std::string toString() const override;
};

class AlarmState : public SoundFaceFollowingState
{
    AlarmManager& m_alarmManager;
    std::string m_alarmPath;

    AlarmStateParameter m_parameter;
    std::optional<uint64_t> m_playSoundDesireId;

public:
    AlarmState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        AlarmManager& alarmManager,
        std::string alarmPath);
    ~AlarmState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(AlarmState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;
};

#endif
