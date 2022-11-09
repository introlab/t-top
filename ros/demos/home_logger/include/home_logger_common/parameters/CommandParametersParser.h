#ifndef HOME_LOGGER_COMMON_PARAMETERS_COMMAND_PARAMETERS_PARSER_H
#define HOME_LOGGER_COMMON_PARAMETERS_COMMAND_PARAMETERS_PARSER_H

#include <home_logger_common/commands/Commands.h>

#include <memory>

class CommandParametersParser
{
public:
    CommandParametersParser();
    virtual ~CommandParametersParser();

    virtual CommandType commandType() const = 0;
    virtual std::shared_ptr<Command> parse(
        const std::shared_ptr<Command>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) = 0;
};


template<class T>
class SpecificCommandParametersParser : public CommandParametersParser
{
public:
    SpecificCommandParametersParser();
    ~SpecificCommandParametersParser() override;

    CommandType commandType() const final;
    std::shared_ptr<Command> parse(
        const std::shared_ptr<Command>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) final;

protected:
    virtual std::shared_ptr<T> parseSpecific(
        const std::shared_ptr<T>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) = 0;
};

template<class T>
SpecificCommandParametersParser<T>::SpecificCommandParametersParser()
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
std::shared_ptr<Command> SpecificCommandParametersParser<T>::parse(
    const std::shared_ptr<Command>& command,
    const std::optional<std::string>& parameterName,
    const std::optional<std::string>& parameterResponse,
    const std::optional<FaceDescriptor>& faceDescriptor)
{
    if ((parameterName.has_value() && !parameterResponse.has_value()) ||
        (!parameterName.has_value() && parameterResponse.has_value()))
    {
        throw std::runtime_error("parameterName and parameterResponse must be set or unset");
    }

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

bool containsAny(const std::string& text, const std::vector<std::string>& keywords);

std::optional<int> findInt(const std::string& text);
std::optional<Time> findTime(const std::string& text);
std::optional<Date> findDate(const std::string& text, int defaultYear, int defaultMonth);
std::optional<int> findWeekDay(const std::string& text);

#endif
