#include "DateTimeCommandExecutors.h"

#include <home_logger_common/DateTime.h>
#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

CurrentDateCommandExecutor::CurrentDateCommandExecutor(StateManager& stateManager)
    : SpecificCommandExecutor<CurrentDateCommand>(stateManager)
{
}

CurrentDateCommandExecutor::~CurrentDateCommandExecutor() {}

void CurrentDateCommandExecutor::executeSpecific(const shared_ptr<CurrentDateCommand>& command)
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        Formatter::format(StringResources::getValue("dialogs.commands.current_date"), fmt::arg("date", Date::now())),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}


CurrentTimeCommandExecutor::CurrentTimeCommandExecutor(StateManager& stateManager)
    : SpecificCommandExecutor<CurrentTimeCommand>(stateManager)
{
}

CurrentTimeCommandExecutor::~CurrentTimeCommandExecutor() {}

void CurrentTimeCommandExecutor::executeSpecific(const shared_ptr<CurrentTimeCommand>& command)
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        Formatter::format(StringResources::getValue("dialogs.commands.current_time"), fmt::arg("time", Time::now())),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}


CurrentDateTimeCommandExecutor::CurrentDateTimeCommandExecutor(StateManager& stateManager)
    : SpecificCommandExecutor<CurrentDateTimeCommand>(stateManager)
{
}

CurrentDateTimeCommandExecutor::~CurrentDateTimeCommandExecutor() {}

void CurrentDateTimeCommandExecutor::executeSpecific(const shared_ptr<CurrentDateTimeCommand>& command)
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        Formatter::format(
            StringResources::getValue("dialogs.commands.current_datetime"),
            fmt::arg("date", Date::now()),
            fmt::arg("time", Time::now())),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}
