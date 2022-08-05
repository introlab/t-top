#ifndef HOME_LOGGER_COMMON_COMMANDS_COMMAND_H
#define HOME_LOGGER_COMMON_COMMANDS_COMMAND_H

#include <home_logger_common/DateTime.h>

#include <tl/optional.hpp>

#include <string>
#include <typeindex>
#include <cstdint>

class CommandType
{
    std::type_index m_type;

    explicit CommandType(std::type_index type);

public:
    template<class T>
    static CommandType get();
    static CommandType null();

    bool operator==(const CommandType& other) const;
    bool operator!=(const CommandType& other) const;
    bool operator<(const CommandType& other) const;
    bool operator<=(const CommandType& other) const;
    bool operator>(const CommandType& other) const;
    bool operator>=(const CommandType& other) const;

    const char* name() const;
    std::size_t hashCode() const;
};

template<class T>
inline CommandType CommandType::get()
{
    return CommandType(std::type_index(typeid(T)));
}

inline CommandType CommandType::null()
{
    return CommandType(std::type_index(typeid(std::nullptr_t)));
}

inline bool CommandType::operator==(const CommandType& other) const
{
    return m_type == other.m_type;
}

inline bool CommandType::operator!=(const CommandType& other) const
{
    return m_type != other.m_type;
}

inline bool CommandType::operator<(const CommandType& other) const
{
    return m_type < other.m_type;
}

inline bool CommandType::operator<=(const CommandType& other) const
{
    return m_type <= other.m_type;
}

inline bool CommandType::operator>(const CommandType& other) const
{
    return m_type > other.m_type;
}

inline bool CommandType::operator>=(const CommandType& other) const
{
    return m_type >= other.m_type;
}

inline const char* CommandType::name() const
{
    return m_type.name();
}

inline std::size_t CommandType::hashCode() const
{
    return m_type.hash_code();
}

namespace std
{
    template<>
    struct hash<CommandType>
    {
        inline std::size_t operator()(const CommandType& type) const { return type.hashCode(); }
    };
}

#define DECLARE_COMMAND_PUBLIC_METHODS(className)                                                                      \
    CommandType type() const override { return CommandType::get<className>(); }

class Command
{
    std::string m_transcript;

public:
    explicit Command(std::string transcript);
    virtual ~Command();

    const std::string& transcript() const;
    virtual CommandType type() const = 0;
};

inline const std::string& Command::transcript() const
{
    return m_transcript;
}


enum class WeatherTime
{
    CURRENT,
    TODAY,
    TOMORROW,
    WEEK
};

class WeatherCommand : public Command
{
    tl::optional<WeatherTime> m_time;

public:
    explicit WeatherCommand(std::string transcript);
    WeatherCommand(std::string transcript, tl::optional<WeatherTime> time);
    ~WeatherCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(WeatherCommand)

    tl::optional<WeatherTime> time() const;
};

inline tl::optional<WeatherTime> WeatherCommand::time() const
{
    return m_time;
}


class IncreaseVolumeCommand : public Command
{
public:
    explicit IncreaseVolumeCommand(std::string transcript);
    ~IncreaseVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(IncreaseVolumeCommand)
};

class DecreaseVolumeCommand : public Command
{
public:
    explicit DecreaseVolumeCommand(std::string transcript);
    ~DecreaseVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(DecreaseVolumeCommand)
};

class MuteCommand : public Command
{
public:
    explicit MuteCommand(std::string transcript);
    ~MuteCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(MuteCommand)
};

class UnmuteCommand : public Command
{
public:
    explicit UnmuteCommand(std::string transcript);
    ~UnmuteCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(UnmuteCommand)
};

class SetVolumeCommand : public Command
{
    tl::optional<float> m_volumePercent;

public:
    explicit SetVolumeCommand(std::string transcript);
    SetVolumeCommand(std::string transcript, tl::optional<float> volumePercent);
    ~SetVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(SetVolumeCommand)

    tl::optional<float> volumePercent() const;
};

inline tl::optional<float> SetVolumeCommand::volumePercent() const
{
    return m_volumePercent;
}

class GetVolumeCommand : public Command
{
public:
    explicit GetVolumeCommand(std::string transcript);
    ~GetVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(GetVolumeCommand)
};


class SleepCommand : public Command
{
public:
    explicit SleepCommand(std::string transcript);
    ~SleepCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(SleepCommand)
};


class CurrentDateCommand : public Command
{
public:
    explicit CurrentDateCommand(std::string transcript);
    ~CurrentDateCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(CurrentDateCommand)
};

class CurrentTimeCommand : public Command
{
public:
    explicit CurrentTimeCommand(std::string transcript);
    ~CurrentTimeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(CurrentTimeCommand)
};

class CurrentDateTimeCommand : public Command
{
public:
    explicit CurrentDateTimeCommand(std::string transcript);
    ~CurrentDateTimeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(CurrentDateTimeCommand)
};


enum class AlarmType
{
    PUNCTUAL,
    REPETTIIVE
};

enum class AlarmFrequency
{
    DAYLY,
    WEEKLY
};

class AddAlarmCommand : public Command
{
    tl::optional<AlarmType> m_alarmType;
    tl::optional<AlarmFrequency> m_frequency;

    tl::optional<int> m_weekDay;
    tl::optional<Date> m_date;
    tl::optional<Time> m_time;

public:
    explicit AddAlarmCommand(std::string transcript);
    AddAlarmCommand(
        std::string transcript,
        tl::optional<AlarmType> alarmType,
        tl::optional<AlarmFrequency> frequency,
        tl::optional<int> weekDay,
        tl::optional<Date> date,
        tl::optional<Time> time);
    ~AddAlarmCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddAlarmCommand)

    tl::optional<AlarmType> alarmType() const;
    tl::optional<AlarmFrequency> frequency() const;

    tl::optional<int> weekDay() const;
    tl::optional<Date> date() const;
    tl::optional<Time> time() const;
};

inline tl::optional<AlarmType> AddAlarmCommand::alarmType() const
{
    return m_alarmType;
}

inline tl::optional<AlarmFrequency> AddAlarmCommand::frequency() const
{
    return m_frequency;
}

inline tl::optional<int> AddAlarmCommand::weekDay() const
{
    return m_weekDay;
}

inline tl::optional<Date> AddAlarmCommand::date() const
{
    return m_date;
}

inline tl::optional<Time> AddAlarmCommand::time() const
{
    return m_time;
}

class ListAlarmsCommand : public Command
{
public:
    explicit ListAlarmsCommand(std::string transcript);
    ~ListAlarmsCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(ListAlarmsCommand)
};

class RemoveAlarmCommand : public Command
{
    tl::optional<int64_t> m_id;

public:
    explicit RemoveAlarmCommand(std::string transcript);
    RemoveAlarmCommand(std::string transcript, tl::optional<int64_t> id);
    ~RemoveAlarmCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(RemoveAlarmCommand)

    tl::optional<int64_t> id() const;
};

inline tl::optional<int64_t> RemoveAlarmCommand::id() const
{
    return m_id;
}


class AddReminderCommand : public Command
{
    tl::optional<std::string> m_text;
    tl::optional<DateTime> m_datetime;

public:
    explicit AddReminderCommand(std::string transcript);
    AddReminderCommand(std::string transcript, tl::optional<std::string> text, tl::optional<DateTime> datetime);
    ~AddReminderCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddReminderCommand)

    tl::optional<std::string> text() const;
    tl::optional<DateTime> datetime() const;
};

inline tl::optional<std::string> AddReminderCommand::text() const
{
    return m_text;
}

inline tl::optional<DateTime> AddReminderCommand::datetime() const
{
    return m_datetime;
}

class ListRemindersCommand : public Command
{
public:
    explicit ListRemindersCommand(std::string transcript);
    ~ListRemindersCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(ListRemindersCommand)
};

class RemoveReminderCommand : public Command
{
    tl::optional<int64_t> m_id;

public:
    explicit RemoveReminderCommand(std::string transcript);
    RemoveReminderCommand(std::string transcript, tl::optional<int64_t> id);
    ~RemoveReminderCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(RemoveReminderCommand)

    tl::optional<int64_t> id() const;
};

inline tl::optional<int64_t> RemoveReminderCommand::id() const
{
    return m_id;
}


class ListCommandsCommand : public Command
{
public:
    explicit ListCommandsCommand(std::string transcript);
    ~ListCommandsCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(ListCommandsCommand)
};

class NothingCommand : public Command
{
public:
    explicit NothingCommand(std::string transcript);
    ~NothingCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(NothingCommand)
};

#endif
