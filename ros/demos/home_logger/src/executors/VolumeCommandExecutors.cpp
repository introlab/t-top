#include "VolumeCommandExecutors.h"

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringRessources.h>

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


MuteCommandExecutor::MuteCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<MuteCommand>(stateManager, volumeManager)
{
}

MuteCommandExecutor::~MuteCommandExecutor() {}

void MuteCommandExecutor::executeSpecific(const shared_ptr<MuteCommand>& command)
{
    m_volumeManager.mute();
    m_stateManager.switchTo<TalkState>(*getAskNextCommandParameter());
}


UnmuteCommandExecutor::UnmuteCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : VolumeCommandExecutor<UnmuteCommand>(stateManager, volumeManager)
{
}

UnmuteCommandExecutor::~UnmuteCommandExecutor() {}

void UnmuteCommandExecutor::executeSpecific(const shared_ptr<UnmuteCommand>& command)
{
    m_volumeManager.unmute();
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
    string text = "";
    if (m_volumeManager.isMuted())
    {
        text = StringRessources::getValue("dialogs.commands.get_volume.muted");
    }
    else
    {
        text = Formatter::format(
            StringRessources::getValue("dialogs.commands.get_volume.volume"),
            fmt::arg("volume", m_volumeManager.getVolume()));
    }

    m_stateManager.switchTo<TalkState>(TalkStateParameter(
        text,
        "",  // No gesture
        "blink",
        StateType::get<TalkState>(),
        getAskNextCommandParameter()));
}
