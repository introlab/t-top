#include "TellReminderState.h"
#include "IdleState.h"

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include <sstream>

using namespace std;

TellReminderStateParameter::TellReminderStateParameter() : reminder("", DateTime(), FaceDescriptor({})) {}

TellReminderStateParameter::TellReminderStateParameter(Reminder reminder) : reminder(move(reminder)) {}

TellReminderStateParameter::~TellReminderStateParameter() {}

string TellReminderStateParameter::toString() const
{
    stringstream ss;
    ss << "text=" << reminder.text();
    return ss.str();
}

TellReminderState::TellReminderState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    ReminderManager& reminderManager)
    : TalkState(stateManager, desireSet, nodeHandle),
      m_reminderManager(reminderManager)
{
}

TellReminderState::~TellReminderState() {}

void TellReminderState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    m_reminder = dynamic_cast<const TellReminderStateParameter&>(parameter).reminder;

    TalkState::onEnabling(
        TalkStateParameter(
            Formatter::format(
                StringResources::getValue("dialogs.tell_reminder_state.reminder"),
                fmt::arg("text", m_reminder.value().text()),
                fmt::arg("time", m_reminder.value().datetime().time)),
            "",  // No gesture
            "blink",
            StateType::get<IdleState>()),
        previousStateType);
}

void TellReminderState::onDisabling()
{
    TalkState::onDisabling();
    if (m_reminder.has_value() && m_reminder.value().id().has_value())
    {
        m_reminderManager.removeReminder(m_reminder.value().id().value());
    }
}
