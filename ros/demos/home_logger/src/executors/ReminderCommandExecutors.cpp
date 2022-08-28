#include "ReminderCommandExecutors.h"

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include <sstream>

using namespace std;

AddReminderCommandExecutor::AddReminderCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager)
    : ReminderCommandExecutor<AddReminderCommand>(stateManager, reminderManager)
{
}

AddReminderCommandExecutor::~AddReminderCommandExecutor() {}

void AddReminderCommandExecutor::executeSpecific(const shared_ptr<AddReminderCommand>& command)
{
    m_reminderManager.insertReminder(Reminder(
        command->text().value(),
        DateTime(command->date().value(), command->time().value()),
        command->faceDescriptor().value()));
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


ListRemindersCommandExecutor::ListRemindersCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager)
    : ReminderCommandExecutor<ListRemindersCommand>(stateManager, reminderManager)
{
}

ListRemindersCommandExecutor::~ListRemindersCommandExecutor() {}

void ListRemindersCommandExecutor::executeSpecific(const shared_ptr<ListRemindersCommand>& command)
{
    stringstream ss;
    auto reminders = m_reminderManager.listReminders();

    if (reminders.empty())
    {
        ss << StringResources::getValue("dialogs.commands.reminder.no_reminder");
    }
    else
    {
        for (const auto& reminder : reminders)
        {
            ss << Formatter::format(
                StringResources::getValue("dialogs.commands.reminder.reminder"),
                fmt::arg("id", reminder.id().value()),
                fmt::arg("text", reminder.text()),
                fmt::arg("date", reminder.datetime().date),
                fmt::arg("time", reminder.datetime().time));
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


RemoveReminderCommandExecutor::RemoveReminderCommandExecutor(
    StateManager& stateManager,
    ReminderManager& reminderManager)
    : ReminderCommandExecutor<RemoveReminderCommand>(stateManager, reminderManager)
{
}

RemoveReminderCommandExecutor::~RemoveReminderCommandExecutor() {}

void RemoveReminderCommandExecutor::executeSpecific(const shared_ptr<RemoveReminderCommand>& command)
{
    m_reminderManager.removeReminder(command->id().value());
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}
