#ifndef HOME_LOGGER_EXECUTORS_VOLUME_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_VOLUME_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"
#include "../managers/VolumeManager.h"

template<class T>
class VolumeCommandExecutor : public SpecificCommandExecutor<T>
{
protected:
    VolumeManager& m_volumeManager;

public:
    VolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager);
    ~VolumeCommandExecutor() override;
};

template<class T>
VolumeCommandExecutor<T>::VolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager)
    : SpecificCommandExecutor<T>(stateManager),
      m_volumeManager(volumeManager)
{
}

template<class T>
VolumeCommandExecutor<T>::~VolumeCommandExecutor()
{
}

class IncreaseVolumeCommandExecutor : public VolumeCommandExecutor<IncreaseVolumeCommand>
{
public:
    IncreaseVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager);
    ~IncreaseVolumeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<IncreaseVolumeCommand>& command) override;
};

class DecreaseVolumeCommandExecutor : public VolumeCommandExecutor<DecreaseVolumeCommand>
{
public:
    DecreaseVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager);
    ~DecreaseVolumeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<DecreaseVolumeCommand>& command) override;
};

class SetVolumeCommandExecutor : public VolumeCommandExecutor<SetVolumeCommand>
{
public:
    SetVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager);
    ~SetVolumeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<SetVolumeCommand>& command) override;
};

class GetVolumeCommandExecutor : public VolumeCommandExecutor<GetVolumeCommand>
{
public:
    GetVolumeCommandExecutor(StateManager& stateManager, VolumeManager& volumeManager);
    ~GetVolumeCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<GetVolumeCommand>& command) override;
};

#endif
