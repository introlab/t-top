#ifndef HOME_LOGGER_COMMON_PARAMETERS_ALARM_COMMAND_PARAMETERS_PARSERS_H
#define HOME_LOGGER_COMMON_PARAMETERS_ALARM_COMMAND_PARAMETERS_PARSERS_H

#include <home_logger_common/commands/Commands.h>
#include <home_logger_common/parameters/CommandParametersParser.h>

class AddAlarmCommandParametersParser : public SpecificCommandParametersParser<AddAlarmCommand>
{
    std::vector<std::string> m_punctualKeywords;
    std::vector<std::string> m_dailyKeywords;
    std::vector<std::string> m_weeklyKeywords;

public:
    AddAlarmCommandParametersParser();
    ~AddAlarmCommandParametersParser() override;

protected:
    std::shared_ptr<AddAlarmCommand> parseSpecific(
        const std::shared_ptr<AddAlarmCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<AddAlarmCommand>
        parseType(const std::shared_ptr<AddAlarmCommand>& command, const std::string& text);
    std::shared_ptr<AddAlarmCommand>
        parseDate(const std::shared_ptr<AddAlarmCommand>& command, const std::string& text);
    std::shared_ptr<AddAlarmCommand>
        parseWeekDay(const std::shared_ptr<AddAlarmCommand>& command, const std::string& text);
    std::shared_ptr<AddAlarmCommand>
        parseTime(const std::shared_ptr<AddAlarmCommand>& command, const std::string& text);
};

class RemoveAlarmCommandParametersParser : public SpecificCommandParametersParser<RemoveAlarmCommand>
{
public:
    RemoveAlarmCommandParametersParser();
    ~RemoveAlarmCommandParametersParser() override;

protected:
    std::shared_ptr<RemoveAlarmCommand> parseSpecific(
        const std::shared_ptr<RemoveAlarmCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<RemoveAlarmCommand>
        parseId(const std::shared_ptr<RemoveAlarmCommand>& command, const std::string& text);
};

#endif
