#ifndef HOME_LOGGER_EXECUTORS_SLEEP_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_SLEEP_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

class SleepCommandExecutor : public SpecificCommandExecutor<SleepCommand>
{
public:
    explicit SleepCommandExecutor(StateManager& stateManager);
    ~SleepCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<SleepCommand>& command) override;
};

#endif
