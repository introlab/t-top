#include "WeatherCommandParametersAsker.h"
#include "../states/common/TalkState.h"
#include "../states/specific/WaitCommandParameterState.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

WeatherCommandParametersAsker::WeatherCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<WeatherCommand>(stateManager)
{
}

WeatherCommandParametersAsker::~WeatherCommandParametersAsker() {}

void WeatherCommandParametersAsker::askSpecific(const shared_ptr<WeatherCommand>& command)
{
    if (!command->time().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.weather.time"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "time")));
    }
    else
    {
        throw runtime_error("The weather command is complete.");
    }
}
