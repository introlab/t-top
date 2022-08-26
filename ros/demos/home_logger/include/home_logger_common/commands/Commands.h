#ifndef HOME_LOGGER_COMMON_COMMANDS_COMMAND_H
#define HOME_LOGGER_COMMON_COMMANDS_COMMAND_H

#include <home_logger_common/DateTime.h>

#include <tl/optional.hpp>

#include <string>
#include <typeindex>
#include <cstdint>
#include <type_traits>
#include <vector>

class Command;

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
    static_assert(std::is_base_of<Command, T>::value, "T must be a subclass of Command");
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
    virtual bool isComplete() const;
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

    bool isComplete() const override;
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

class SetVolumeCommand : public Command
{
    tl::optional<float> m_volumePercent;

public:
    explicit SetVolumeCommand(std::string transcript);
    SetVolumeCommand(std::string transcript, tl::optional<float> volumePercent);
    ~SetVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(SetVolumeCommand)

    bool isComplete() const override;
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
    DAILY,
    WEEKLY,
};

class AddAlarmCommand : public Command
{
    tl::optional<AlarmType> m_alarmType;

    tl::optional<int> m_weekDay;
    tl::optional<Date> m_date;
    tl::optional<Time> m_time;

public:
    explicit AddAlarmCommand(std::string transcript);
    AddAlarmCommand(
        std::string transcript,
        tl::optional<AlarmType> alarmType,
        tl::optional<int> weekDay,
        tl::optional<Date> date,
        tl::optional<Time> time);
    ~AddAlarmCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddAlarmCommand)

    bool isComplete() const override;

    tl::optional<AlarmType> alarmType() const;

    tl::optional<int> weekDay() const;
    tl::optional<Date> date() const;
    tl::optional<Time> time() const;
};

inline tl::optional<AlarmType> AddAlarmCommand::alarmType() const
{
    return m_alarmType;
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

    bool isComplete() const override;
    tl::optional<int64_t> id() const;
};

inline tl::optional<int64_t> RemoveAlarmCommand::id() const
{
    return m_id;
}


class FaceDescriptor
{
    std::vector<float> m_descriptor;

public:
    FaceDescriptor(std::vector<float> descriptor);
    virtual ~FaceDescriptor();

    const std::vector<float>& data() const;

    float distance(const FaceDescriptor& other) const;
    static FaceDescriptor mean(const std::vector<FaceDescriptor>& descriptors);
};

inline const std::vector<float>& FaceDescriptor::data() const
{
    return m_descriptor;
}

class AddReminderCommand : public Command
{
    tl::optional<std::string> m_text;
    tl::optional<DateTime> m_datetime;
    tl::optional<FaceDescriptor> m_faceDescriptor;

public:
    explicit AddReminderCommand(std::string transcript);
    AddReminderCommand(std::string transcript, tl::optional<std::string> text, tl::optional<DateTime> datetime, tl::optional<FaceDescriptor> faceDescriptor);
    ~AddReminderCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddReminderCommand)

    bool isComplete() const override;
    tl::optional<std::string> text() const;
    tl::optional<DateTime> datetime() const;
    tl::optional<FaceDescriptor> faceDescriptor() const;
};

inline tl::optional<std::string> AddReminderCommand::text() const
{
    return m_text;
}

inline tl::optional<DateTime> AddReminderCommand::datetime() const
{
    return m_datetime;
}

inline tl::optional<FaceDescriptor> AddReminderCommand::faceDescriptor() const
{
    return m_faceDescriptor;
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

    bool isComplete() const override;
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
