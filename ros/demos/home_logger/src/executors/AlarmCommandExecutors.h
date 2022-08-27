#ifndef HOME_LOGGER_EXECUTORS_ALARM_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_ALARM_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

#include <home_logger_common/managers/AlarmManager.h>

template<class T>
class AlarmCommandExecutor : public SpecificCommandExecutor<T>
{
protected:
    AlarmManager& m_alarmManager;

public:
    AlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager);
    ~AlarmCommandExecutor() override;
};

template<class T>
AlarmCommandExecutor<T>::AlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager)
    : SpecificCommandExecutor<T>(stateManager),
      m_alarmManager(alarmManager)
{
}

template<class T>
AlarmCommandExecutor<T>::~AlarmCommandExecutor()
{
}

class AddAlarmCommandExecutor : public AlarmCommandExecutor<AddAlarmCommand>
{
public:
    AddAlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager);
    ~AddAlarmCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<AddAlarmCommand>& command) override;
};

class ListAlarmsCommandExecutor : public AlarmCommandExecutor<ListAlarmsCommand>
{
public:
    ListAlarmsCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager);
    ~ListAlarmsCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<ListAlarmsCommand>& command) override;
};

class RemoveAlarmCommandExecutor : public AlarmCommandExecutor<RemoveAlarmCommand>
{
public:
    RemoveAlarmCommandExecutor(StateManager& stateManager, AlarmManager& alarmManager);
    ~RemoveAlarmCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<RemoveAlarmCommand>& command) override;
};

#endif
