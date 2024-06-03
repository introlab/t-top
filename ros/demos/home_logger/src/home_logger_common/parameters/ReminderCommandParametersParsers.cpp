#include <home_logger_common/parameters/ReminderCommandParametersParsers.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

AddReminderCommandParametersParser::AddReminderCommandParametersParser() {}

AddReminderCommandParametersParser::~AddReminderCommandParametersParser() {}

shared_ptr<AddReminderCommand> AddReminderCommandParametersParser::parseSpecific(
    const shared_ptr<AddReminderCommand>& command,
    const optional<string>& parameterName,
    const optional<string>& parameterResponse,
    const optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        return parseFaceDescriptor(command, faceDescriptor.value());
    }

    if (!parameterName.has_value())
    {
        return command;
    }
    else if (parameterName == "text")
    {
        return parseText(command, parameterResponse.value());
    }
    else if (parameterName == "date")
    {
        return parseDate(command, parameterResponse.value());
    }
    else if (parameterName == "time")
    {
        return parseTime(command, parameterResponse.value());
    }
    else
    {
        throw runtime_error(
            "AddReminderCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<AddReminderCommand>
    AddReminderCommandParametersParser::parseText(const shared_ptr<AddReminderCommand>& command, const string& text)
{
    return make_shared<AddReminderCommand>(
        command->transcript(),
        text,
        command->date(),
        command->time(),
        command->faceDescriptor());
}

shared_ptr<AddReminderCommand>
    AddReminderCommandParametersParser::parseDate(const shared_ptr<AddReminderCommand>& command, const string& text)
{
    auto now = Date::now();
    return make_shared<AddReminderCommand>(
        command->transcript(),
        command->text(),
        findDate(text, now.year, now.month),
        command->time(),
        command->faceDescriptor());
}

shared_ptr<AddReminderCommand>
    AddReminderCommandParametersParser::parseTime(const shared_ptr<AddReminderCommand>& command, const string& text)
{
    return make_shared<AddReminderCommand>(
        command->transcript(),
        command->text(),
        command->date(),
        findTime(text),
        command->faceDescriptor());
}

shared_ptr<AddReminderCommand> AddReminderCommandParametersParser::parseFaceDescriptor(
    const shared_ptr<AddReminderCommand>& command,
    const FaceDescriptor& faceDescriptor)
{
    return make_shared<AddReminderCommand>(
        command->transcript(),
        command->text(),
        command->date(),
        command->time(),
        faceDescriptor);
}


RemoveReminderCommandParametersParser::RemoveReminderCommandParametersParser() {}

RemoveReminderCommandParametersParser::~RemoveReminderCommandParametersParser() {}

shared_ptr<RemoveReminderCommand> RemoveReminderCommandParametersParser::parseSpecific(
    const shared_ptr<RemoveReminderCommand>& command,
    const optional<string>& parameterName,
    const optional<string>& parameterResponse,
    const optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        throw runtime_error("RemoveReminderCommandParametersParser doesn't support faceDescriptor");
    }

    if (!parameterName.has_value())
    {
        return parseId(command, command->transcript());
    }
    else if (parameterName == "id")
    {
        return parseId(command, parameterResponse.value());
    }
    else
    {
        throw runtime_error(
            "RemoveReminderCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<RemoveReminderCommand>
    RemoveReminderCommandParametersParser::parseId(const shared_ptr<RemoveReminderCommand>& command, const string& text)
{
    return make_shared<RemoveReminderCommand>(command->transcript(), findInt(text));
}
