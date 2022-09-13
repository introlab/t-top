#ifndef HOME_LOGGER_PARAMETERS_COMMAND_PARAMETERS_ASKER_H
#define HOME_LOGGER_PARAMETERS_COMMAND_PARAMETERS_ASKER_H

#include "../states/StateManager.h"

#include <home_logger_common/commands/Commands.h>

#include <memory>

class CommandParametersAsker
{
protected:
    StateManager& m_stateManager;

public:
    explicit CommandParametersAsker(StateManager& stateManager);
    virtual ~CommandParametersAsker();

    virtual CommandType commandType() const = 0;
    virtual void ask(const std::shared_ptr<Command>& command) = 0;
};


template<class T>
class SpecificCommandParametersAsker : public CommandParametersAsker
{
public:
    explicit SpecificCommandParametersAsker(StateManager& stateManager);
    ~SpecificCommandParametersAsker() override;

    CommandType commandType() const final;
    void ask(const std::shared_ptr<Command>& command) final;

protected:
    virtual void askSpecific(const std::shared_ptr<T>& command) = 0;
};

template<class T>
SpecificCommandParametersAsker<T>::SpecificCommandParametersAsker(StateManager& stateManager)
    : CommandParametersAsker(stateManager)
{
}

template<class T>
SpecificCommandParametersAsker<T>::~SpecificCommandParametersAsker()
{
}

template<class T>
CommandType SpecificCommandParametersAsker<T>::commandType() const
{
    return CommandType::get<T>();
}

template<class T>
void SpecificCommandParametersAsker<T>::ask(const std::shared_ptr<Command>& command)
{
    std::shared_ptr<T> specificCommand = std::dynamic_pointer_cast<T>(command);
    if (specificCommand)
    {
        askSpecific(specificCommand);
    }
    else
    {
        throw std::runtime_error("Invalid command parameters asker");
    }
}

#endif
