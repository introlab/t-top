#include <home_logger_common/commands/AllCommandParser.h>
#include <home_logger_common/language/StringResources.h>


using namespace std;

AllCommandParser::AllCommandParser()
{
    m_parsers.emplace_back(make_unique<KeywordCommandParser<WeatherCommand>>(
        vector<SynonymKeywords>({StringResources::getVector("weather_command.weather")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<IncreaseVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("increase_volume_command.increase"),
             StringResources::getVector("increase_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<DecreaseVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("decrease_volume_command.decrease"),
             StringResources::getVector("decrease_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<SetVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("set_volume_command.set"),
             StringResources::getVector("set_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<GetVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("get_volume_command.get"),
             StringResources::getVector("get_volume_command.volume")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<SleepCommand>>(
        vector<SynonymKeywords>({StringResources::getVector("sleep_command.sleep")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentDateCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("current_date_command.what"),
             StringResources::getVector("current_date_command.date")}),
        StringResources::getVector("current_date_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentTimeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("current_time_command.what"),
             StringResources::getVector("current_time_command.time")}),
        StringResources::getVector("current_time_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentDateTimeCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("current_date_time_command.what"),
             StringResources::getVector("current_date_time_command.date"),
             StringResources::getVector("current_date_time_command.time")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<AddAlarmCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("add_alarm_command.add"),
             StringResources::getVector("add_alarm_command.alarm")}),
        StringResources::getVector("add_alarm_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListAlarmsCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("list_alarms_command.list"),
             StringResources::getVector("list_alarms_command.alarm")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<RemoveAlarmCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("remove_alarm_command.remove"),
             StringResources::getVector("remove_alarm_command.alarm")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<AddReminderCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("add_reminder_command.add"),
             StringResources::getVector("add_reminder_command.reminder")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListRemindersCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("list_reminders_command.list"),
             StringResources::getVector("list_reminders_command.reminder")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<RemoveReminderCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("remove_reminder_command.remove"),
             StringResources::getVector("remove_reminder_command.reminder")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListCommandsCommand>>(
        vector<SynonymKeywords>(
            {StringResources::getVector("list_commands_command.list"),
             StringResources::getVector("list_commands_command.commands")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<NothingCommand>>(
        vector<SynonymKeywords>({StringResources::getVector("nothing_command.nothing")}),
        vector<string>({})));
}

AllCommandParser::~AllCommandParser() {}

vector<unique_ptr<Command>> AllCommandParser::parse(const string& transcript)
{
    vector<unique_ptr<Command>> commands;

    for (auto& parser : m_parsers)
    {
        unique_ptr<Command> command = parser->parse(transcript);
        if (command != nullptr)
        {
            commands.emplace_back(move(command));
        }
    }

    return commands;
}
