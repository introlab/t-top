#ifndef HOME_LOGGER_COMMON_PARAMETERS_REMINDER_COMMAND_PARAMETERS_PARSERS_H
#define HOME_LOGGER_COMMON_PARAMETERS_REMINDER_COMMAND_PARAMETERS_PARSERS_H

#include <home_logger_common/commands/Commands.h>
#include <home_logger_common/parameters/CommandParametersParser.h>

class AddReminderCommandParametersParser : public SpecificCommandParametersParser<AddReminderCommand>
{
public:
    AddReminderCommandParametersParser();
    ~AddReminderCommandParametersParser() override;

protected:
    std::shared_ptr<AddReminderCommand> parseSpecific(
        const std::shared_ptr<AddReminderCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<AddReminderCommand>
        parseText(const std::shared_ptr<AddReminderCommand>& command, const std::string& text);
    std::shared_ptr<AddReminderCommand>
        parseDate(const std::shared_ptr<AddReminderCommand>& command, const std::string& text);
    std::shared_ptr<AddReminderCommand>
        parseTime(const std::shared_ptr<AddReminderCommand>& command, const std::string& text);
    std::shared_ptr<AddReminderCommand>
        parseFaceDescriptor(const std::shared_ptr<AddReminderCommand>& command, const FaceDescriptor& faceDescriptor);
};

class RemoveReminderCommandParametersParser : public SpecificCommandParametersParser<RemoveReminderCommand>
{
public:
    RemoveReminderCommandParametersParser();
    ~RemoveReminderCommandParametersParser() override;

protected:
    std::shared_ptr<RemoveReminderCommand> parseSpecific(
        const std::shared_ptr<RemoveReminderCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<RemoveReminderCommand>
        parseId(const std::shared_ptr<RemoveReminderCommand>& command, const std::string& text);
};

#endif
