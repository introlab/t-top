#ifndef HOME_LOGGER_EXECUTOR_ALL_COMMAND_EXECUTOR_H
#define HOME_LOGGER_EXECUTOR_ALL_COMMAND_EXECUTOR_H

#include "CommandExecutor.h"
#include "../managers/VolumeManager.h"

#include <unordered_map>
#include <memory>

class AllCommandExecutor
{
    std::unordered_map<CommandType, std::unique_ptr<CommandExecutor>> m_commandExecutorByCommandType;

public:
    AllCommandExecutor(StateManager& stateManager, ros::NodeHandle& nodeHandle, VolumeManager& volumeManager);
    virtual ~AllCommandExecutor();

    virtual void execute(const std::shared_ptr<Command>& command);
};

#endif
