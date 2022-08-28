#include "ListCommandsCommandExecutor.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

ListCommandsCommandExecutor::ListCommandsCommandExecutor(StateManager& stateManager)
    : SpecificCommandExecutor<ListCommandsCommand>(stateManager)
{
}

ListCommandsCommandExecutor::~ListCommandsCommandExecutor() {}

void ListCommandsCommandExecutor::executeSpecific(const shared_ptr<ListCommandsCommand>& command)
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        StringResources::getValue("dialogs.commands.list_commands"),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}
