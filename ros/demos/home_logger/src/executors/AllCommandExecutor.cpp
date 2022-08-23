#include "AllCommandExecutor.h"

#include "WeatherCommandExecutor.h"
#include "VolumeCommandExecutors.h"
#include "DateTimeCommandExecutors.h"
#include "SleepCommandExecutor.h"
#include "ListCommandsCommandExecutor.h"

using namespace std;

AllCommandExecutor::AllCommandExecutor(
    StateManager& stateManager,
    ros::NodeHandle& nodeHandle,
    VolumeManager& volumeManager)
{
    vector<unique_ptr<CommandExecutor>> executors;

    executors.emplace_back(make_unique<WeatherCommandExecutor>(stateManager, nodeHandle));

    executors.emplace_back(make_unique<IncreaseVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<DecreaseVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<MuteCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<UnmuteCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<SetVolumeCommandExecutor>(stateManager, volumeManager));
    executors.emplace_back(make_unique<GetVolumeCommandExecutor>(stateManager, volumeManager));

    executors.emplace_back(make_unique<SleepCommandExecutor>(stateManager));

    executors.emplace_back(make_unique<CurrentDateCommandExecutor>(stateManager));
    executors.emplace_back(make_unique<CurrentTimeCommandExecutor>(stateManager));
    executors.emplace_back(make_unique<CurrentDateTimeCommandExecutor>(stateManager));

    executors.emplace_back(make_unique<ListCommandsCommandExecutor>(stateManager));


    for (auto& executor : executors)
    {
        if (m_commandExecutorByCommandType.find(executor->commandType()) != m_commandExecutorByCommandType.end())
        {
            throw runtime_error(
                string("The executor for ") + executor->commandType().name() + " commands is already declared.");
        }
        m_commandExecutorByCommandType[executor->commandType()] = move(executor);
    }
}

AllCommandExecutor::~AllCommandExecutor() {}

void AllCommandExecutor::execute(const std::shared_ptr<Command>& command)
{
    auto it = m_commandExecutorByCommandType.find(command->type());
    if (it == m_commandExecutorByCommandType.end())
    {
        throw runtime_error(command->type().name() + string(" type does not have any executor."));
    }

    it->second->execute(command);
}
