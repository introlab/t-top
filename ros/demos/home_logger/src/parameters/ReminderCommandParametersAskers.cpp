#include "ReminderCommandParametersAskers.h"
#include "../states/common/TalkState.h"
#include "../states/specific/WaitCommandParameterState.h"
#include "../states/specific/WaitFaceDescriptorCommandParameterState.h"

#include <home_logger_common/language/StringResources.h>

using namespace std;

AddReminderCommandParametersAsker::AddReminderCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<AddReminderCommand>(stateManager)
{
}

AddReminderCommandParametersAsker::~AddReminderCommandParametersAsker() {}

void AddReminderCommandParametersAsker::askSpecific(const shared_ptr<AddReminderCommand>& command)
{
    if (!command->text().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_reminder.text"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "text")));
    }
    else if (!command->date().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_reminder.date"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "date")));
    }
    else if (!command->time().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_reminder.time"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "time")));
    }
    else if (!command->faceDescriptor().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.add_reminder.face_descriptor"),
            "",  // No gesture
            "blink",
            StateType::get<WaitFaceDescriptorCommandParameterState>(),
            make_shared<WaitFaceDescriptorCommandParameterStateParameter>(command)));
    }
    else
    {
        throw runtime_error("The add reminder command is complete.");
    }
}

RemoveReminderCommandParametersAsker::RemoveReminderCommandParametersAsker(StateManager& stateManager)
    : SpecificCommandParametersAsker<RemoveReminderCommand>(stateManager)
{
}

RemoveReminderCommandParametersAsker::~RemoveReminderCommandParametersAsker() {}

void RemoveReminderCommandParametersAsker::askSpecific(const shared_ptr<RemoveReminderCommand>& command)
{
    if (!command->id().has_value())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.command_parameters.remove_reminder.id"),
            "",  // No gesture
            "blink",
            StateType::get<WaitCommandParameterState>(),
            make_shared<WaitCommandParameterStateParameter>(command, "id")));
    }
    else
    {
        throw runtime_error("The remove reminder command is complete.");
    }
}
