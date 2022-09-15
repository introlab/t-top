#include "AllCommandExecutor.h"

#include "WeatherCommandExecutor.h"
#include "VolumeCommandExecutors.h"
#include "SleepCommandExecutor.h"
#include "DateTimeCommandExecutors.h"
#include "AlarmCommandExecutors.h"
#include "ReminderCommandExecutors.h"
#include "ListCommandsCommandExecutor.h"

using namespace std;

AllCommandExecutor::AllCommandExecutor(
    StateManager& stateManager,
    ros::NodeHandle& nodeHandle,
    VolumeManager& volumeManager,
    AlarmManager& alarmManager,
    ReminderManager& reminderManager)
{
    vector<unique_ptr<CommandExecutor>> executors;

    executors.emplace_back(make_unique<WeatherCommandExecutor>(stateManager, nodeHandle));

    executors.emplace_back(make_unique<IncreaseVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<DecreaseVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<SetVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<GetVolumeCommandExecutor>(stateManager, volumeManager));

    executors.emplace_back(make_unique<SleepCommandExecutor>(stateManager));

    executors.emplace_back(make_unique<CurrentDateCommandExecutor>(stateManager));
    executors.emplace_back(make_unique<CurrentTimeCommandExecutor>(stateManager));
    executors.emplace_back(make_unique<CurrentDateTimeCommandExecutor>(stateManager));

    executors.emplace_back(make_unique<AddAlarmCommandExecutor>(stateManager, alarmManager));
    executors.emplace_back(make_unique<ListAlarmsCommandExecutor>(stateManager, alarmManager));
    executors.emplace_back(make_unique<RemoveAlarmCommandExecutor>(stateManager, alarmManager));

    executors.emplace_back(make_unique<AddReminderCommandExecutor>(stateManager, reminderManager));
    executors.emplace_back(make_unique<ListRemindersCommandExecutor>(stateManager, reminderManager));
    executors.emplace_back(make_unique<RemoveReminderCommandExecutor>(stateManager, reminderManager));

    executors.emplace_back(make_unique<ListCommandsCommandExecutor>(stateManager));


    for (auto& executor : executors)
    {
        if (m_commandExecutorsByCommandType.find(executor->commandType()) != m_commandExecutorsByCommandType.end())
        {
            throw runtime_error(
                string("The executor for ") + executor->commandType().name() + " commands is already declared.");
        }
        m_commandExecutorsByCommandType[executor->commandType()] = move(executor);
    }
}

AllCommandExecutor::~AllCommandExecutor() {}

void AllCommandExecutor::execute(const shared_ptr<Command>& command)
{
    auto it = m_commandExecutorsByCommandType.find(command->type());
    if (it == m_commandExecutorsByCommandType.end())
    {
        throw runtime_error(command->type().name() + string(" type does not have any executor."));
    }

    it->second->execute(command);
}
