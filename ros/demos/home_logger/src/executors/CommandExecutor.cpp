#include "CommandExecutor.h"

#include "../states/specific/WaitCommandState.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

CommandExecutor::CommandExecutor(StateManager& stateManager) : m_stateManager(stateManager) {}

CommandExecutor::~CommandExecutor() {}

shared_ptr<TalkStateParameter> CommandExecutor::getAskNextCommandParameter()
{
    return make_shared<TalkStateParameter>(
        StringResources::getValue("dialogs.execute_command_state.ask_next_command"),
        "",  // No gesture
        "blink",
        StateType::get<WaitCommandState>());
}
