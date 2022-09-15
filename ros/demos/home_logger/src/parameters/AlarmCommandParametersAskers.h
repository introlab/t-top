#ifndef HOME_LOGGER_PARAMETERS_ALARM_COMMAND_PARAMETERS_ASKERS_H
#define HOME_LOGGER_PARAMETERS_ALARM_COMMAND_PARAMETERS_ASKERS_H

#include "CommandParametersAsker.h"

#include <home_logger_common/commands/Commands.h>

class AddAlarmCommandParametersAsker : public SpecificCommandParametersAsker<AddAlarmCommand>
{
public:
    explicit AddAlarmCommandParametersAsker(StateManager& stateManager);
    ~AddAlarmCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<AddAlarmCommand>& command) override;
};

class RemoveAlarmCommandParametersAsker : public SpecificCommandParametersAsker<RemoveAlarmCommand>
{
public:
    explicit RemoveAlarmCommandParametersAsker(StateManager& stateManager);
    ~RemoveAlarmCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<RemoveAlarmCommand>& command) override;
};

#endif
