#include <home_logger_common/parameters/AllCommandParametersParser.h>

#include <home_logger_common/parameters/WeatherCommandParametersParser.h>
#include <home_logger_common/parameters/SetVolumeCommandParametersParser.h>
#include <home_logger_common/parameters/AlarmCommandParametersParsers.h>
#include <home_logger_common/parameters/ReminderCommandParametersParsers.h>

using namespace std;

AllCommandParametersParser::AllCommandParametersParser()
{
    vector<unique_ptr<CommandParametersParser>> parsers;

    parsers.emplace_back(make_unique<WeatherCommandParametersParser>());

    parsers.emplace_back(make_unique<SetVolumeCommandParametersParser>());

    parsers.emplace_back(make_unique<AddAlarmCommandParametersParser>());
    parsers.emplace_back(make_unique<RemoveAlarmCommandParametersParser>());

    parsers.emplace_back(make_unique<AddReminderCommandParametersParser>());
    parsers.emplace_back(make_unique<RemoveReminderCommandParametersParser>());


    for (auto& parser : parsers)
    {
        if (m_commandParameterParsersByCommandType.find(parser->commandType()) !=
            m_commandParameterParsersByCommandType.end())
        {
            throw runtime_error(
                string("The parser for ") + parser->commandType().name() + " commands is already declared.");
        }
        m_commandParameterParsersByCommandType[parser->commandType()] = move(parser);
    }
}

AllCommandParametersParser::~AllCommandParametersParser() {}

shared_ptr<Command> AllCommandParametersParser::parse(
    const shared_ptr<Command>& command,
    const std::optional<string>& parameterName,
    const std::optional<string>& parameterResponse,
    const std::optional<FaceDescriptor>& faceDescriptor)
{
    auto it = m_commandParameterParsersByCommandType.find(command->type());
    if (it == m_commandParameterParsersByCommandType.end())
    {
        throw runtime_error(command->type().name() + string(" type does not have any command parameter parser."));
    }

    return it->second->parse(command, parameterName, parameterResponse, faceDescriptor);
}
