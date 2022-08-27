#ifndef HOME_LOGGER_EXECUTORS_REMINDER_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_REMINDER_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

#include <home_logger_common/managers/ReminderManager.h>

template<class T>
class ReminderCommandExecutor : public SpecificCommandExecutor<T>
{
protected:
    ReminderManager& m_reminderManager;

public:
    ReminderCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager);
    ~ReminderCommandExecutor() override;
};

template<class T>
ReminderCommandExecutor<T>::ReminderCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager)
    : SpecificCommandExecutor<T>(stateManager),
      m_reminderManager(reminderManager)
{
}

template<class T>
ReminderCommandExecutor<T>::~ReminderCommandExecutor()
{
}

class AddReminderCommandExecutor : public ReminderCommandExecutor<AddReminderCommand>
{
public:
    AddReminderCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager);
    ~AddReminderCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<AddReminderCommand>& command) override;
};

class ListRemindersCommandExecutor : public ReminderCommandExecutor<ListRemindersCommand>
{
public:
    ListRemindersCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager);
    ~ListRemindersCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<ListRemindersCommand>& command) override;
};

class RemoveReminderCommandExecutor : public ReminderCommandExecutor<RemoveReminderCommand>
{
public:
    RemoveReminderCommandExecutor(StateManager& stateManager, ReminderManager& reminderManager);
    ~RemoveReminderCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<RemoveReminderCommand>& command) override;
};

#endif
