#ifndef HOME_LOGGER_EXECUTORS_DATE_TIME_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_DATE_TIME_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

class CurrentDateCommandExecutor : public SpecificCommandExecutor<CurrentDateCommand>
{
public:
    explicit CurrentDateCommandExecutor(StateManager& stateManager);
    ~CurrentDateCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentDateCommand>& command) override;
};

class CurrentTimeCommandExecutor : public SpecificCommandExecutor<CurrentTimeCommand>
{
public:
    explicit CurrentTimeCommandExecutor(StateManager& stateManager);
    ~CurrentTimeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentTimeCommand>& command) override;
};

class CurrentDateTimeCommandExecutor : public SpecificCommandExecutor<CurrentDateTimeCommand>
{
public:
    explicit CurrentDateTimeCommandExecutor(StateManager& stateManager);
    ~CurrentDateTimeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<CurrentDateTimeCommand>& command) override;
};

#endif
