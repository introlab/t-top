#ifndef HOME_LOGGER_STATES_SPECIFIC_EXECUTE_COMMAND_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_EXECUTE_COMMAND_STATE_H

#include "../common/SoundFaceFollowingState.h"
#include "../../managers/VolumeManager.h"
#include "../../executors/AllCommandExecutor.h"

class ExecuteCommandStateParameter : public StateParameter
{
public:
    std::shared_ptr<Command> command;

    ExecuteCommandStateParameter();
    ExecuteCommandStateParameter(std::shared_ptr<Command> command);
    ~ExecuteCommandStateParameter() override;

    std::string toString() const;
};

class ExecuteCommandState : public SoundFaceFollowingState
{
    AllCommandExecutor m_allCommandExecutor;

public:
    ExecuteCommandState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        VolumeManager& volumeManager);
    ~ExecuteCommandState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(ExecuteCommandState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
};

#endif
