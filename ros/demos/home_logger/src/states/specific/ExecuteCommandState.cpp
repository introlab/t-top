#include "ExecuteCommandState.h"
#include "WaitCommandState.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringResources.h>

#include <sstream>
#include <vector>

using namespace std;

ExecuteCommandStateParameter::ExecuteCommandStateParameter() {}

ExecuteCommandStateParameter::ExecuteCommandStateParameter(shared_ptr<Command> command) : command(move(command)) {}

ExecuteCommandStateParameter::ExecuteCommandStateParameter(
    shared_ptr<Command> command,
    string parameterName,
    string parameterResponse)
    : command(command),
      parameterName(move(parameterName)),
      parameterResponse(move(parameterResponse))
{
}

ExecuteCommandStateParameter::ExecuteCommandStateParameter(shared_ptr<Command> command, FaceDescriptor faceDescriptor)
    : command(move(command)),
      faceDescriptor(move(faceDescriptor))
{
}

ExecuteCommandStateParameter::~ExecuteCommandStateParameter() {}

string ExecuteCommandStateParameter::toString() const
{
    stringstream ss;
    ss << "command_type=" << command->type().name();
    return ss.str();
}


ExecuteCommandState::ExecuteCommandState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    VolumeManager& volumeManager,
    AlarmManager& alarmManager,
    ReminderManager& reminderManager)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_allCommandExecutor(stateManager, nodeHandle, volumeManager, alarmManager, reminderManager),
      m_allCommandParametersAsker(stateManager)
{
}

ExecuteCommandState::~ExecuteCommandState() {}

void ExecuteCommandState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    auto executeCommandParameter = dynamic_cast<const ExecuteCommandStateParameter&>(parameter);

    shared_ptr<Command> command = executeCommandParameter.command;
    if (!command->isComplete())
    {
        command = m_allCommandParametersParser.parse(
            command,
            executeCommandParameter.parameterName,
            executeCommandParameter.parameterResponse,
            executeCommandParameter.faceDescriptor);
    }

    if (command->isComplete() && previousStateType == StateType::get<WaitCommandState>())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.execute_command_state.valid_command"),
            "yes",
            "happy",
            StateType::get<ExecuteCommandState>(),
            make_shared<ExecuteCommandStateParameter>(executeCommandParameter)));
    }
    else if (command->isComplete())
    {
        m_allCommandExecutor.execute(command);
    }
    else
    {
        m_allCommandParametersAsker.ask(command);
    }
}
