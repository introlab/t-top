#ifndef HOME_LOGGER_STATES_SPECIFIC_EXECUTE_COMMAND_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_EXECUTE_COMMAND_STATE_H

#include "../common/SoundFaceFollowingState.h"
#include "../../managers/VolumeManager.h"
#include "../../executors/AllCommandExecutor.h"
#include "../../parameters/AllCommandParametersAsker.h"

#include <home_logger_common/parameters/AllCommandParametersParser.h>

class ExecuteCommandStateParameter : public StateParameter
{
public:
    std::shared_ptr<Command> command;

    std::optional<std::string> parameterName;
    std::optional<std::string> parameterResponse;
    std::optional<FaceDescriptor> faceDescriptor;

    ExecuteCommandStateParameter();
    explicit ExecuteCommandStateParameter(std::shared_ptr<Command> command);
    ExecuteCommandStateParameter(
        std::shared_ptr<Command> command,
        std::string parameterName,
        std::string parameterResponse);
    ExecuteCommandStateParameter(std::shared_ptr<Command> command, FaceDescriptor faceDescriptor);
    ~ExecuteCommandStateParameter() override;

    std::string toString() const override;
};

class ExecuteCommandState : public SoundFaceFollowingState
{
    AllCommandExecutor m_allCommandExecutor;
    AllCommandParametersAsker m_allCommandParametersAsker;
    AllCommandParametersParser m_allCommandParametersParser;

public:
    ExecuteCommandState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        VolumeManager& volumeManager,
        AlarmManager& alarmManager,
        ReminderManager& reminderManager);
    ~ExecuteCommandState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(ExecuteCommandState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
};

#endif
