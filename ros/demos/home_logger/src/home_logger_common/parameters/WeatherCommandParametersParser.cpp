#include <home_logger_common/parameters/WeatherCommandParametersParser.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

WeatherCommandParametersParser::WeatherCommandParametersParser()
{
    m_currentKeywords = StringResources::getVector("weather_command.time.current");
    m_todayKeywords = StringResources::getVector("weather_command.time.today");
    m_tomorrowKeywords = StringResources::getVector("weather_command.time.tomorrow");
    m_weekKeywords = StringResources::getVector("weather_command.time.week");
}

WeatherCommandParametersParser::~WeatherCommandParametersParser() {}

shared_ptr<WeatherCommand> WeatherCommandParametersParser::parseSpecific(
    const shared_ptr<WeatherCommand>& command,
    const tl::optional<string>& parameterName,
    const tl::optional<string>& parameterResponse,
    const tl::optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        throw runtime_error("WeatherCommandParametersParser doesn't support faceDescriptor");
    }

    if (!parameterName.has_value())
    {
        return parseTime(command, command->transcript());
    }
    else if (parameterName == "time")
    {
        return parseTime(command, parameterResponse.value());
    }
    else
    {
        throw runtime_error(
            "WeatherCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<WeatherCommand>
    WeatherCommandParametersParser::parseTime(const shared_ptr<WeatherCommand>& command, const string& text)
{
    string lowerCaseText = toLowerString(text);
    tl::optional<WeatherTime> time;
    if (containsAny(lowerCaseText, m_currentKeywords))
    {
        time = WeatherTime::CURRENT;
    }
    else if (containsAny(lowerCaseText, m_todayKeywords))
    {
        time = WeatherTime::TODAY;
    }
    else if (containsAny(lowerCaseText, m_tomorrowKeywords))
    {
        time = WeatherTime::TOMORROW;
    }
    else if (containsAny(lowerCaseText, m_weekKeywords))
    {
        time = WeatherTime::WEEK;
    }

    return make_shared<WeatherCommand>(command->transcript(), time);
}
