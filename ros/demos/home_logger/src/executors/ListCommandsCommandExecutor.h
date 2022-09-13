#ifndef HOME_LOGGER_EXECUTORS_LIST_COMMANDS_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_LIST_COMMANDS_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

class ListCommandsCommandExecutor : public SpecificCommandExecutor<ListCommandsCommand>
{
public:
    explicit ListCommandsCommandExecutor(StateManager& stateManager);
    ~ListCommandsCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<ListCommandsCommand>& command) override;
};

#endif
