#ifndef HOME_LOGGER_PARAMETERS_REMINDER_COMMAND_PARAMETERS_ASKERS_H
#define HOME_LOGGER_PARAMETERS_REMINDER_COMMAND_PARAMETERS_ASKERS_H

#include "CommandParametersAsker.h"

#include <home_logger_common/commands/Commands.h>

class AddReminderCommandParametersAsker : public SpecificCommandParametersAsker<AddReminderCommand>
{
public:
    explicit AddReminderCommandParametersAsker(StateManager& stateManager);
    ~AddReminderCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<AddReminderCommand>& command) override;
};

class RemoveReminderCommandParametersAsker : public SpecificCommandParametersAsker<RemoveReminderCommand>
{
public:
    explicit RemoveReminderCommandParametersAsker(StateManager& stateManager);
    ~RemoveReminderCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<RemoveReminderCommand>& command) override;
};

#endif
