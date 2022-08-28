#include "VolumeCommandExecutors.h"

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

IncreaseVolumeCommandExecutor::IncreaseVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<IncreaseVolumeCommand>(stateManager, volumeManager)
{
}

IncreaseVolumeCommandExecutor::~IncreaseVolumeCommandExecutor() {}

void IncreaseVolumeCommandExecutor::executeSpecific(const shared_ptr<IncreaseVolumeCommand>& command)
{
    m_volumeManager.setVolume(m_volumeManager.getVolume() + 5.f);
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


DecreaseVolumeCommandExecutor::DecreaseVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<DecreaseVolumeCommand>(stateManager, volumeManager)
{
}

DecreaseVolumeCommandExecutor::~DecreaseVolumeCommandExecutor() {}

void DecreaseVolumeCommandExecutor::executeSpecific(const shared_ptr<DecreaseVolumeCommand>& command)
{
    m_volumeManager.setVolume(m_volumeManager.getVolume() - 5.f);
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


SetVolumeCommandExecutor::SetVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<SetVolumeCommand>(stateManager, volumeManager)
{
}

SetVolumeCommandExecutor::~SetVolumeCommandExecutor() {}

void SetVolumeCommandExecutor::executeSpecific(const shared_ptr<SetVolumeCommand>& command)
{
    m_volumeManager.setVolume(command->volumePercent().value());
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


GetVolumeCommandExecutor::GetVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<GetVolumeCommand>(stateManager, volumeManager)
{
}

GetVolumeCommandExecutor::~GetVolumeCommandExecutor() {}

void GetVolumeCommandExecutor::executeSpecific(const shared_ptr<GetVolumeCommand>& command)
{
    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        Formatter::format(
            StringResources::getValue("dialogs.commands.get_volume.volume"),
            fmt::arg("volume", m_volumeManager.getVolume())),
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}
