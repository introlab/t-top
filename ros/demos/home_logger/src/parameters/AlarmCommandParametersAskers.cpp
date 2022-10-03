#include "AlarmCommandParametersAskers.h"
#include "../states/common/TalkState.h"
#include "../states/specific/WaitCommandParameterState.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

AddAlarmCommandParametersAsker::AddAlarmCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<AddAlarmCommand>(stateManager)
{
}

AddAlarmCommandParametersAsker::~AddAlarmCommandParametersAsker() {}

void AddAlarmCommandParametersAsker::askSpecific(const shared_ptr<AddAlarmCommand>& command)
{
    if (!command->alarmType().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_alarm.type"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "type")));
    }
    else if (command->alarmType() == AlarmType::PUNCTUAL && !command->date().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_alarm.date"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "date")));
    }
    else if (command->alarmType() == AlarmType::WEEKLY && !command->weekDay().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_alarm.week_day"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "week_day")));
    }
    else if (!command->time().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_alarm.time"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "time")));
    }
    else
    {
        throw runtime_error("The add alarm command is complete.");
    }
}

RemoveAlarmCommandParametersAsker::RemoveAlarmCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<RemoveAlarmCommand>(stateManager)
{
}

RemoveAlarmCommandParametersAsker::~RemoveAlarmCommandParametersAsker() {}

void RemoveAlarmCommandParametersAsker::askSpecific(const shared_ptr<RemoveAlarmCommand>& command)
{
    if (!command->id().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.remove_alarm.id"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "id")));
    }
    else
    {
        throw runtime_error("The remove alarm command is complete.");
    }
}
