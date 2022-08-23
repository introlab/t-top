#include <home_logger_common/commands/AllCommandParser.h>
#include <home_logger_common/language/StringRessources.h>


using namespace std;

AllCommandParser::AllCommandParser()
{
    m_parsers.emplace_back(make_unique<KeywordCommandParser<WeatherCommand>>(
        vector<SynonymKeywords>({StringRessources::getVector("weather_command.weather")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<IncreaseVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("increase_volume_command.increase"),
             StringRessources::getVector("increase_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<DecreaseVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("decrease_volume_command.decrease"),
             StringRessources::getVector("decrease_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<MuteCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("mute_command.set"), StringRessources::getVector("mute_command.mute")}),
        StringRessources::getVector("mute_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<UnmuteCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("unmute_command.unset"),
             StringRessources::getVector("unmute_command.unmute")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<SetVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("set_volume_command.set"),
             StringRessources::getVector("set_volume_command.volume")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<GetVolumeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("get_volume_command.get"),
             StringRessources::getVector("get_volume_command.volume")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<SleepCommand>>(
        vector<SynonymKeywords>({StringRessources::getVector("sleep_command.sleep")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentDateCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("current_date_command.what"),
             StringRessources::getVector("current_date_command.date")}),
        StringRessources::getVector("current_date_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentTimeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("current_time_command.what"),
             StringRessources::getVector("current_time_command.time")}),
        StringRessources::getVector("current_time_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<CurrentDateTimeCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("current_date_time_command.what"),
             StringRessources::getVector("current_date_time_command.date"),
             StringRessources::getVector("current_date_time_command.time")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<AddAlarmCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("add_alarm_command.add"),
             StringRessources::getVector("add_alarm_command.alarm")}),
        StringRessources::getVector("add_alarm_command.not_keywords")));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListAlarmsCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("list_alarms_command.list"),
             StringRessources::getVector("list_alarms_command.alarm")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<RemoveAlarmCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("remove_alarm_command.remove"),
             StringRessources::getVector("remove_alarm_command.alarm")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<AddReminderCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("add_reminder_command.add"),
             StringRessources::getVector("add_reminder_command.reminder")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListRemindersCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("list_reminders_command.list"),
             StringRessources::getVector("list_reminders_command.reminder")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<RemoveReminderCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("remove_reminder_command.remove"),
             StringRessources::getVector("remove_reminder_command.reminder")}),
        vector<string>({})));

    m_parsers.emplace_back(make_unique<KeywordCommandParser<ListCommandsCommand>>(
        vector<SynonymKeywords>(
            {StringRessources::getVector("list_commands_command.list"),
             StringRessources::getVector("list_commands_command.commands")}),
        vector<string>({})));
    m_parsers.emplace_back(make_unique<KeywordCommandParser<NothingCommand>>(
        vector<SynonymKeywords>({StringRessources::getVector("nothing_command.nothing")}),
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
