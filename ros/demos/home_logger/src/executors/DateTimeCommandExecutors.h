#ifndef HOME_LOGGER_EXECUTOR_DATE_TIME_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTOR_DATE_TIME_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

class CurrentDateCommandExecutor : public SpecificCommandExecutor<CurrentDateCommand>
{
public:
    CurrentDateCommandExecutor(StateManager& stateManager);
    ~CurrentDateCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentDateCommand>& command) override;
};

class CurrentTimeCommandExecutor : public SpecificCommandExecutor<CurrentTimeCommand>
{
public:
    CurrentTimeCommandExecutor(StateManager& stateManager);
    ~CurrentTimeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentTimeCommand>& command) override;
};

class CurrentDateTimeCommandExecutor : public SpecificCommandExecutor<CurrentDateTimeCommand>
{
public:
    CurrentDateTimeCommandExecutor(StateManager& stateManager);
    ~CurrentDateTimeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentDateTimeCommand>& command) override;
};

#endif
