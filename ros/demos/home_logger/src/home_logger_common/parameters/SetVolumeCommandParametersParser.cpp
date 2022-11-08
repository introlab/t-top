#include <home_logger_common/parameters/SetVolumeCommandParametersParser.h>

using namespace std;

SetVolumeCommandParametersParser::SetVolumeCommandParametersParser() {}

SetVolumeCommandParametersParser::~SetVolumeCommandParametersParser() {}

shared_ptr<SetVolumeCommand> SetVolumeCommandParametersParser::parseSpecific(
    const shared_ptr<SetVolumeCommand>& command,
    const std::optional<string>& parameterName,
    const std::optional<string>& parameterResponse,
    const std::optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        throw runtime_error("SetVolumeCommandParametersParser doesn't support faceDescriptor");
    }

    if (!parameterName.has_value())
    {
        return parseVolume(command, command->transcript());
    }
    else if (parameterName == "volume")
    {
        return parseVolume(command, parameterResponse.value());
    }
    else
    {
        throw runtime_error(
            "SetVolumeCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<SetVolumeCommand>
    SetVolumeCommandParametersParser::parseVolume(const shared_ptr<SetVolumeCommand>& command, const string& text)
{
    std::optional<float> volume = findInt(text).map([](int v) { return static_cast<float>(v); });
    return make_shared<SetVolumeCommand>(command->transcript(), volume);
}
