#ifndef HOME_LOGGER_COMMON_PARAMETERS_WEATHER_COMMAND_PARAMETERS_PARSER_H
#define HOME_LOGGER_COMMON_PARAMETERS_WEATHER_COMMAND_PARAMETERS_PARSER_H

#include <home_logger_common/commands/Commands.h>
#include <home_logger_common/parameters/CommandParametersParser.h>

class WeatherCommandParametersParser : public SpecificCommandParametersParser<WeatherCommand>
{
    std::vector<std::string> m_currentKeywords;
    std::vector<std::string> m_todayKeywords;
    std::vector<std::string> m_tomorrowKeywords;
    std::vector<std::string> m_weekKeywords;

public:
    WeatherCommandParametersParser();
    ~WeatherCommandParametersParser() override;

protected:
    std::shared_ptr<WeatherCommand> parseSpecific(
        const std::shared_ptr<WeatherCommand>& command,
        const std::optional<std::string>& parameterName,
        const std::optional<std::string>& parameterResponse,
        const std::optional<FaceDescriptor>& faceDescriptor) override;

private:
    std::shared_ptr<WeatherCommand> parseTime(const std::shared_ptr<WeatherCommand>& command, const std::string& text);
};

#endif
