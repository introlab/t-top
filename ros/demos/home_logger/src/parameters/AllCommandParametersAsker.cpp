#include "AllCommandParametersAsker.h"

#include "WeatherCommandParametersAsker.h"
#include "SetVolumeCommandParametersAsker.h"
#include "AlarmCommandParametersAskers.h"
#include "ReminderCommandParametersAskers.h"

using namespace std;

AllCommandParametersAsker::AllCommandParametersAsker(StateManager& stateManager)
{
    vector<unique_ptr<CommandParametersAsker>> askers;

    askers.emplace_back(make_unique<WeatherCommandParametersAsker>(stateManager));

    askers.emplace_back(make_unique<SetVolumeCommandParametersAsker>(stateManager));

    askers.emplace_back(make_unique<AddAlarmCommandParametersAsker>(stateManager));
    askers.emplace_back(make_unique<RemoveAlarmCommandParametersAsker>(stateManager));

    askers.emplace_back(make_unique<AddReminderCommandParametersAsker>(stateManager));
    askers.emplace_back(make_unique<RemoveReminderCommandParametersAsker>(stateManager));


    for (auto& asker : askers)
    {
        if (m_commandParameterAskersByCommandType.find(asker->commandType()) !=
            m_commandParameterAskersByCommandType.end())
        {
            throw runtime_error(
                string("The asker for ") + asker->commandType().name() + " commands is already declared.");
        }
        m_commandParameterAskersByCommandType[asker->commandType()] = move(asker);
    }
}

AllCommandParametersAsker::~AllCommandParametersAsker() {}

void AllCommandParametersAsker::ask(const shared_ptr<Command>& command)
{
    auto it = m_commandParameterAskersByCommandType.find(command->type());
    if (it == m_commandParameterAskersByCommandType.end())
    {
        throw runtime_error(command->type().name() + string(" type does not have any command parameter asker."));
    }

    it->second->ask(command);
}
