#ifndef HOME_LOGGER_EXECUTORS_ALL_COMMAND_EXECUTOR_H
#define HOME_LOGGER_EXECUTORS_ALL_COMMAND_EXECUTOR_H

#include "CommandExecutor.h"
#include "../managers/VolumeManager.h"

#include <home_logger_common/managers/AlarmManager.h>
#include <home_logger_common/managers/ReminderManager.h>

#include <unordered_map>
#include <memory>

class AllCommandExecutor
{
    std::unordered_map<CommandType, std::unique_ptr<CommandExecutor>> m_commandExecutorsByCommandType;

public:
    AllCommandExecutor(
        StateManager& stateManager,
        ros::NodeHandle& nodeHandle,
        VolumeManager& volumeManager,
        AlarmManager& alarmManager,
        ReminderManager& reminderManager);
    virtual ~AllCommandExecutor();

    void execute(const std::shared_ptr<Command>& command);
};

#endif
