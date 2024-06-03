#ifndef HOME_LOGGER_PARAMETERS_ALL_COMMAND_PARAMETES_ASKER_H
#define HOME_LOGGER_PARAMETERS_ALL_COMMAND_PARAMETES_ASKER_H

#include "CommandParametersAsker.h"

#include <unordered_map>
#include <memory>

class AllCommandParametersAsker
{
    std::unordered_map<CommandType, std::unique_ptr<CommandParametersAsker>> m_commandParameterAskersByCommandType;

public:
    explicit AllCommandParametersAsker(StateManager& stateManager);
    virtual ~AllCommandParametersAsker();

    void ask(const std::shared_ptr<Command>& command);
};

#endif
