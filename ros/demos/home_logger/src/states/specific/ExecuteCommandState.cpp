#include "ExecuteCommandState.h"
#include "WaitCommandState.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringRessources.h>

#include <sstream>
#include <vector>

using namespace std;

ExecuteCommandStateParameter::ExecuteCommandStateParameter() {}

ExecuteCommandStateParameter::ExecuteCommandStateParameter(shared_ptr<Command> command) : command(command) {}

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
    AlarmManager& alarmManager)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_allCommandExecutor(stateManager, nodeHandle, volumeManager, alarmManager)
{
}

ExecuteCommandState::~ExecuteCommandState() {}

void ExecuteCommandState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    auto executeCommandParameter = dynamic_cast<const ExecuteCommandStateParameter&>(parameter);

    if (executeCommandParameter.command->isComplete() && previousStateType == StateType::get<WaitCommandState>())
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringRessources::getValue("dialogs.execute_command_state.valid_command"),
            "yes",
            "happy",
            StateType::get<ExecuteCommandState>(),
            make_shared<ExecuteCommandStateParameter>(executeCommandParameter)));
    }
    else if (executeCommandParameter.command->isComplete())
    {
        m_allCommandExecutor.execute(executeCommandParameter.command);
    }
    else
    {
        // TODO parameter parsers/questions
        // WeatherCommand
        // SetVolumeCommand
        // AddAlarmCommand
        // RemoveAlarmCommand
        // AddReminderCommand
        // RemoveReminderCommand
    }
}
