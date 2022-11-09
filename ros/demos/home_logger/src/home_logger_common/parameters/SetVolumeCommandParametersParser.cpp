#include <home_logger_common/parameters/SetVolumeCommandParametersParser.h>

using namespace std;

SetVolumeCommandParametersParser::SetVolumeCommandParametersParser() {}

SetVolumeCommandParametersParser::~SetVolumeCommandParametersParser() {}

shared_ptr<SetVolumeCommand> SetVolumeCommandParametersParser::parseSpecific(
    const shared_ptr<SetVolumeCommand>& command,
    const optional<string>& parameterName,
    const optional<string>& parameterResponse,
    const optional<FaceDescriptor>& faceDescriptor)
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
    optional<int> volumeInt = findInt(text);
    optional<float> volumeFloat;
    if (volumeInt.has_value())
    {
        volumeFloat = static_cast<float>(volumeInt.value());
    }

    return make_shared<SetVolumeCommand>(command->transcript(), volumeFloat);
}
