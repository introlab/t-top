#ifndef HOME_LOGGER_EXECUTORS_COMMAND_EXECUTOR_H
#define HOME_LOGGER_EXECUTORS_COMMAND_EXECUTOR_H

#include "../states/StateManager.h"
#include "../states/common/TalkState.h"

#include <home_logger_common/commands/Commands.h>

#include <memory>

class CommandExecutor
{
protected:
    StateManager& m_stateManager;

public:
    explicit CommandExecutor(StateManager& stateManager);
    virtual ~CommandExecutor();

    virtual CommandType commandType() const = 0;
    virtual void execute(const std::shared_ptr<Command>& command) = 0;

protected:
    std::shared_ptr<TalkStateParameter> getAskNextCommandParameter();
};


template<class T>
class SpecificCommandExecutor : public CommandExecutor
{
public:
    explicit SpecificCommandExecutor(StateManager& stateManager);
    ~SpecificCommandExecutor() override;

    CommandType commandType() const final;
    void execute(const std::shared_ptr<Command>& command) final;

protected:
    virtual void executeSpecific(const std::shared_ptr<T>& command) = 0;
};

template<class T>
SpecificCommandExecutor<T>::SpecificCommandExecutor(StateManager& stateManager) : CommandExecutor(stateManager)
{
}

template<class T>
SpecificCommandExecutor<T>::~SpecificCommandExecutor()
{
}

template<class T>
CommandType SpecificCommandExecutor<T>::commandType() const
{
    return CommandType::get<T>();
}

template<class T>
void SpecificCommandExecutor<T>::execute(const std::shared_ptr<Command>& command)
{
    std::shared_ptr<T> specificCommand = std::dynamic_pointer_cast<T>(command);
    if (specificCommand && specificCommand->isComplete())
    {
        executeSpecific(specificCommand);
    }
    else
    {
        throw std::runtime_error("Invalid command executor or incomplet command");
    }
}

#endif
