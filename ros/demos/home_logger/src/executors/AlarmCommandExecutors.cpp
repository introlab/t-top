#include "AlarmCommandExecutors.h"

#include <home_logger_common/language/StringResources.h>

#include <sstream>

using namespace std;

AddAlarmCommandExecutor::AddAlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager)
    : AlarmCommandExecutor<AddAlarmCommand>(stateManager, alarmManager)
{
}

AddAlarmCommandExecutor::~AddAlarmCommandExecutor() {}

void AddAlarmCommandExecutor::executeSpecific(const shared_ptr<AddAlarmCommand>& command)
{
    m_alarmManager.insertAlarm(toAlarm(*command));
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


ListAlarmsCommandExecutor::ListAlarmsCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager)
    : AlarmCommandExecutor<ListAlarmsCommand>(stateManager, alarmManager)
{
}

ListAlarmsCommandExecutor::~ListAlarmsCommandExecutor() {}

void ListAlarmsCommandExecutor::executeSpecific(const shared_ptr<ListAlarmsCommand>& command)
{
    stringstream ss;
    auto alarms = m_alarmManager.listAlarms();

    if (alarms.empty())
    {
        ss << StringResources::getValue("dialogs.commands.alarm.no_alarm");
    }
    else
    {
        for (const auto& alarm : alarms)
        {
            ss << alarm->toSpeech();
            ss << "\n";
        }
    }

    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        ss.str(),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}


RemoveAlarmCommandExecutor::RemoveAlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager)
    : AlarmCommandExecutor<RemoveAlarmCommand>(stateManager, alarmManager)
{
}

RemoveAlarmCommandExecutor::~RemoveAlarmCommandExecutor() {}

void RemoveAlarmCommandExecutor::executeSpecific(const shared_ptr<RemoveAlarmCommand>& command)
{
    m_alarmManager.removeAlarm(command->id().value());
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}
