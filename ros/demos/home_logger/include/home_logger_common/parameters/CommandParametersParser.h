#ifndef HOME_LOGGER_COMMON_PARAMETERS_COMMAND_PARAMETERS_PARSER_H
#define HOME_LOGGER_COMMON_PARAMETERS_COMMAND_PARAMETERS_PARSER_H

#include "../states/StateManager.h"

#include <home_logger_common/commands/Commands.h>

#include <memory>

class CommandParametersParser
{
protected:
    StateManager& m_stateManager;

public:
    CommandParametersParser(StateManager& stateManager);
    virtual ~CommandParametersParser();

    virtual CommandType commandType() const = 0;
    virtual void ask(const std::shared_ptr<Command>& command) = 0;
};


template<class T>
class SpecificCommandParametersParser : public CommandParametersParser
{
public:
    SpecificCommandParametersParser(StateManager& stateManager);
    ~SpecificCommandParametersParser() override;

    CommandType commandType() const final;
    std::shared_ptr<Command> parse(
        std::shared_ptr<Command>& command,
        const tl::optional<std::string>& parameterName,
        const tl::optional<std::string>& parameterResponse,
        const tl::optional<FaceDescriptor>& faceDescriptor) final;

protected:
    virtual std::shared_ptr<T> parseSpecific(
        const std::shared_ptr<T>& command,
        const tl::optional<std::string>& parameterName,
        const tl::optional<std::string>& parameterResponse,
        const tl::optional<FaceDescriptor>& faceDescriptor) = 0;
};

template<class T>
SpecificCommandParametersParser<T>::SpecificCommandParametersParser(StateManager& stateManager)
    : CommandParametersParser(stateManager)
{
}

template<class T>
SpecificCommandParametersParser<T>::~SpecificCommandParametersParser()
{
}

template<class T>
CommandType SpecificCommandParametersParser<T>::commandType() const
{
    return CommandType::get<T>();
}

template<class T>
void SpecificCommandParametersParser<T>::parse(
    std::shared_ptr<Command>& command,
    const tl::optional<std::string>& parameterName,
    const tl::optional<std::string>& parameterResponse,
    const tl::optional<FaceDescriptor>& faceDescriptor)
{
    std::shared_ptr<T> specificCommand = std::dynamic_pointer_cast<T>(command);
    if (specificCommand)
    {
        return parseSpecific(specificCommand, parameterName, parameterResponse, faceDescriptor);
    }
    else
    {
        throw std::runtime_error("Invalid command parameters parser");
    }
}

#endif
