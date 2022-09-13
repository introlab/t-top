#ifndef HOME_LOGGER_PARAMETERS_SET_VOLUME_COMMAND_PARAMETERS_ASKER_H
#define HOME_LOGGER_PARAMETERS_SET_VOLUME_COMMAND_PARAMETERS_ASKER_H

#include "CommandParametersAsker.h"

#include <home_logger_common/commands/Commands.h>

class SetVolumeCommandParametersAsker : public SpecificCommandParametersAsker<SetVolumeCommand>
{
public:
    explicit SetVolumeCommandParametersAsker(StateManager& stateManager);
    ~SetVolumeCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<SetVolumeCommand>& command) override;
};

#endif
