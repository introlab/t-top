#ifndef HOME_LOGGER_PARAMETERS_WEATHER_COMMAND_PARAMETERS_ASKER_H
#define HOME_LOGGER_PARAMETERS_WEATHER_COMMAND_PARAMETERS_ASKER_H

#include "CommandParametersAsker.h"

#include <home_logger_common/commands/Commands.h>

class WeatherCommandParametersAsker : public SpecificCommandParametersAsker<WeatherCommand>
{
public:
    explicit WeatherCommandParametersAsker(StateManager& stateManager);
    ~WeatherCommandParametersAsker() override;

protected:
    void askSpecific(const std::shared_ptr<WeatherCommand>& command) override;
};

#endif
