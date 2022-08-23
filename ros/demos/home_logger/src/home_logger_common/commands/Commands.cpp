#include <home_logger_common/commands/Commands.h>

using namespace std;

CommandType::CommandType(std::type_index type) : m_type(type) {}

Command::Command(string transcript) : m_transcript(move(transcript)) {}

Command::~Command() {}

bool Command::isComplete() const
{
    return true;
}


WeatherCommand::WeatherCommand(string transcript) : Command(move(transcript)) {}

WeatherCommand::WeatherCommand(string transcript, tl::optional<WeatherTime> time)
    : Command(move(transcript)),
      m_time(time)
{
}

WeatherCommand::~WeatherCommand() {}

bool WeatherCommand::isComplete() const
{
    return m_time.has_value();
}


IncreaseVolumeCommand::IncreaseVolumeCommand(string transcript) : Command(move(transcript)) {}

IncreaseVolumeCommand::~IncreaseVolumeCommand() {}

DecreaseVolumeCommand::DecreaseVolumeCommand(string transcript) : Command(move(transcript)) {}

DecreaseVolumeCommand::~DecreaseVolumeCommand() {}

SetVolumeCommand::SetVolumeCommand(string transcript) : Command(move(transcript)) {}

SetVolumeCommand::SetVolumeCommand(string transcript, tl::optional<float> volumePercent)
    : Command(move(transcript)),
      m_volumePercent(volumePercent)
{
}

bool SetVolumeCommand::isComplete() const
{
    return m_volumePercent.has_value();
}

GetVolumeCommand::GetVolumeCommand(string transcript) : Command(move(transcript)) {}

GetVolumeCommand::~GetVolumeCommand() {}

SetVolumeCommand::~SetVolumeCommand() {}


SleepCommand::SleepCommand(string transcript) : Command(move(transcript)) {}

SleepCommand::~SleepCommand() {}


CurrentDateCommand::CurrentDateCommand(string transcript) : Command(move(transcript)) {}

CurrentDateCommand::~CurrentDateCommand() {}

CurrentTimeCommand::CurrentTimeCommand(string transcript) : Command(move(transcript)) {}

CurrentTimeCommand::~CurrentTimeCommand() {}

CurrentDateTimeCommand::CurrentDateTimeCommand(string transcript) : Command(move(transcript)) {}

CurrentDateTimeCommand::~CurrentDateTimeCommand() {}


AddAlarmCommand::AddAlarmCommand(std::string transcript) : Command(move(transcript)) {}

AddAlarmCommand::AddAlarmCommand(
    std::string transcript,
    tl::optional<AlarmType> alarmType,
    tl::optional<int> weekDay,
    tl::optional<Date> date,
    tl::optional<Time> time)
    : Command(move(transcript)),
      m_alarmType(alarmType),
      m_weekDay(weekDay),
      m_date(date),
      m_time(time)
{
}

AddAlarmCommand::~AddAlarmCommand() {}

bool AddAlarmCommand::isComplete() const
{
    if (!m_alarmType.has_value())
    {
        return false;
    }
    if (m_alarmType == AlarmType::PUNCTUAL)
    {
        return m_date.has_value() && m_time.has_value();
    }

    if (m_alarmType == AlarmType::DAYLY)
    {
        return m_time.has_value();
    }
    else
    {
        return m_time.has_value() && m_weekDay.has_value();
    }

    return false;
}

ListAlarmsCommand::ListAlarmsCommand(std::string transcript) : Command(move(transcript)) {}

ListAlarmsCommand::~ListAlarmsCommand() {}

RemoveAlarmCommand::RemoveAlarmCommand(std::string transcript) : Command(move(transcript)) {}

RemoveAlarmCommand::RemoveAlarmCommand(std::string transcript, tl::optional<int64_t> id)
    : Command(move(transcript)),
      m_id(id)
{
}

RemoveAlarmCommand::~RemoveAlarmCommand() {}

bool RemoveAlarmCommand::isComplete() const
{
    return m_id.has_value();
}


AddReminderCommand::AddReminderCommand(std::string transcript) : Command(move(transcript)) {}

AddReminderCommand::AddReminderCommand(
    std::string transcript,
    tl::optional<std::string> text,
    tl::optional<DateTime> datetime)
    : Command(move(transcript)),
      m_text(text),
      m_datetime(datetime)
{
}

AddReminderCommand::~AddReminderCommand() {}

bool AddReminderCommand::isComplete() const
{
    return m_text.has_value() && m_datetime.has_value();
}

ListRemindersCommand::ListRemindersCommand(std::string transcript) : Command(move(transcript)) {}

ListRemindersCommand::~ListRemindersCommand() {}

RemoveReminderCommand::RemoveReminderCommand(std::string transcript) : Command(move(transcript)) {}

RemoveReminderCommand::RemoveReminderCommand(std::string transcript, tl::optional<int64_t> id)
    : Command(move(transcript)),
      m_id(id)
{
}

RemoveReminderCommand::~RemoveReminderCommand() {}

bool RemoveReminderCommand::isComplete() const
{
    return m_id.has_value();
}


ListCommandsCommand::ListCommandsCommand(std::string transcript) : Command(move(transcript)) {}

ListCommandsCommand::~ListCommandsCommand() {}

NothingCommand::NothingCommand(std::string transcript) : Command(move(transcript)) {}

NothingCommand::~NothingCommand() {}
