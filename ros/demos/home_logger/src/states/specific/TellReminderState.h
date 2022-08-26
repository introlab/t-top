#ifndef HOME_LOGGER_STATES_SPECIFIC_TELL_REMINDER_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_TELL_REMINDER_STATE_H

#include "../common/TalkState.h"

#include <home_logger_common/DateTime.h>
#include <home_logger_common/managers/ReminderManager.h>

class TellReminderStateParameter : public StateParameter
{
public:
    Reminder reminder;

    TellReminderStateParameter();
    TellReminderStateParameter(Reminder reminder);
    ~TellReminderStateParameter() override;

    std::string toString() const;
};

class TellReminderState : public TalkState
{
    ReminderManager& m_reminderManager;
    tl::optional<Reminder> m_reminder;

public:
    TellReminderState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        ReminderManager& reminderManager);
    ~TellReminderState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(TellReminderState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;
};

#endif
