#include "SetVolumeCommandParametersAsker.h"
#include "../states/common/TalkState.h"
#include "../states/specific/WaitCommandParameterState.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

SetVolumeCommandParametersAsker::SetVolumeCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<SetVolumeCommand>(stateManager)
{
}

SetVolumeCommandParametersAsker::~SetVolumeCommandParametersAsker() {}

void SetVolumeCommandParametersAsker::askSpecific(const shared_ptr<SetVolumeCommand>& command)
{
    if (!command->volumePercent().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.set_volume.volume"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "volume")));
    }
    else
    {
        throw runtime_error("The set volume command is complete.");
    }
}
