#include <home_logger_common/parameters/AlarmCommandParametersParsers.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

AddAlarmCommandParametersParser::AddAlarmCommandParametersParser()
{
    m_punctualKeywords = StringResources::getVector("add_alarm_command.punctual");
    m_dailyKeywords = StringResources::getVector("add_alarm_command.daily");
    m_weeklyKeywords = StringResources::getVector("add_alarm_command.weekly");
}

AddAlarmCommandParametersParser::~AddAlarmCommandParametersParser() {}

shared_ptr<AddAlarmCommand> AddAlarmCommandParametersParser::parseSpecific(
    const shared_ptr<AddAlarmCommand>& command,
    const optional<string>& parameterName,
    const optional<string>& parameterResponse,
    const optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        throw runtime_error("AddAlarmCommandParametersParser doesn't support faceDescriptor");
    }

    if (!parameterName.has_value())
    {
        return parseType(command, command->transcript());
    }
    else if (parameterName == "type")
    {
        return parseType(command, parameterResponse.value());
    }
    else if (parameterName == "date")
    {
        return parseDate(command, parameterResponse.value());
    }
    else if (parameterName == "week_day")
    {
        return parseWeekDay(command, parameterResponse.value());
    }
    else if (parameterName == "time")
    {
        return parseTime(command, parameterResponse.value());
    }
    else
    {
        throw runtime_error(
            "AddAlarmCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<AddAlarmCommand>
    AddAlarmCommandParametersParser::parseType(const shared_ptr<AddAlarmCommand>& command, const string& text)
{
    string lowerCaseText = toLowerString(text);
    optional<AlarmType> alarmType;
    if (containsAny(lowerCaseText, m_punctualKeywords))
    {
        alarmType = AlarmType::PUNCTUAL;
    }
    else if (containsAny(lowerCaseText, m_dailyKeywords))
    {
        alarmType = AlarmType::DAILY;
    }
    else if (containsAny(lowerCaseText, m_weeklyKeywords))
    {
        alarmType = AlarmType::WEEKLY;
    }

    return make_shared<AddAlarmCommand>(
        command->transcript(),
        alarmType,
        command->weekDay(),
        command->date(),
        command->time());
}

shared_ptr<AddAlarmCommand>
    AddAlarmCommandParametersParser::parseDate(const shared_ptr<AddAlarmCommand>& command, const string& text)
{
    auto now = Date::now();
    return make_shared<AddAlarmCommand>(
        command->transcript(),
        command->alarmType(),
        command->weekDay(),
        findDate(text, now.year, now.month),
        command->time());
}

shared_ptr<AddAlarmCommand>
    AddAlarmCommandParametersParser::parseWeekDay(const shared_ptr<AddAlarmCommand>& command, const string& text)
{
    return make_shared<AddAlarmCommand>(
        command->transcript(),
        command->alarmType(),
        findWeekDay(text),
        command->date(),
        command->time());
}

shared_ptr<AddAlarmCommand>
    AddAlarmCommandParametersParser::parseTime(const shared_ptr<AddAlarmCommand>& command, const string& text)
{
    return make_shared<AddAlarmCommand>(
        command->transcript(),
        command->alarmType(),
        command->weekDay(),
        command->date(),
        findTime(text));
}


RemoveAlarmCommandParametersParser::RemoveAlarmCommandParametersParser() {}

RemoveAlarmCommandParametersParser::~RemoveAlarmCommandParametersParser() {}

shared_ptr<RemoveAlarmCommand> RemoveAlarmCommandParametersParser::parseSpecific(
    const shared_ptr<RemoveAlarmCommand>& command,
    const optional<string>& parameterName,
    const optional<string>& parameterResponse,
    const optional<FaceDescriptor>& faceDescriptor)
{
    if (faceDescriptor.has_value())
    {
        throw runtime_error("RemoveAlarmCommandParametersParser doesn't support faceDescriptor");
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
            "RemoveAlarmCommandParametersParser doesn't support the parameter (" + parameterName.value() + ")");
    }
}

shared_ptr<RemoveAlarmCommand>
    RemoveAlarmCommandParametersParser::parseId(const shared_ptr<RemoveAlarmCommand>& command, const string& text)
{
    return make_shared<RemoveAlarmCommand>(command->transcript(), findInt(text));
}
