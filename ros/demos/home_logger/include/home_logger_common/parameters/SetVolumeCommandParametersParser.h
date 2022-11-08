#ifndef HOME_LOGGER_COMMON_PARAMETERS_SET_VOLUME_COMMAND_PARAMETERS_PARSER_H
#define HOME_LOGGER_COMMON_PARAMETERS_SET_VOLUME_COMMAND_PARAMETERS_PARSER_H

#include <home_logger_common/commands/Commands.h>
#include <home_logger_common/parameters/CommandParametersParser.h>

class SetVolumeCommandParametersParser : public SpecificCommandParametersParser<SetVolumeCommand>
{
public:
    SetVolumeCommandParametersParser();
    ~SetVolumeCommandParametersParser() override;

protected:
    std::shared_ptr<SetVolumeCommand> parseSpecific(
        const std::shared_ptr<SetVolumeCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<SetVolumeCommand>
        parseVolume(const std::shared_ptr<SetVolumeCommand>& command, const std::string& text);
};

#endif
