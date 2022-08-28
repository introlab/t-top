#ifndef HOME_LOGGER_COMMON_PARAMETERS_ALL_COMMAND_PARAMETERS_PARSER_H
#define HOME_LOGGER_COMMON_PARAMETERS_ALL_COMMAND_PARAMETERS_PARSER_H

#include <home_logger_common/parameters/CommandParametersParser.h>

#include <unordered_map>
#include <memory>

class AllCommandParametersParser
{
    std::unordered_map<CommandType, std::unique_ptr<CommandParametersParser>> m_commandParameterParsersByCommandType;

public:
    AllCommandParametersParser();
    virtual ~AllCommandParametersParser();

    std::shared_ptr<Command> parse(
        const std::shared_ptr<Command>& command,
        const tl::optional<std::string>& parameterName,
        const tl::optional<std::string>& parameterResponse,
        const tl::optional<FaceDescriptor>& faceDescriptor);
};

#endif
