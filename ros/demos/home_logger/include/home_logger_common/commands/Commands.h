#ifndef HOME_LOGGER_COMMON_COMMANDS_COMMAND_H
#define HOME_LOGGER_COMMON_COMMANDS_COMMAND_H

#include <home_logger_common/DateTime.h>

#include <optional>
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
    std::optional<WeatherTime> m_time;

public:
    explicit WeatherCommand(std::string transcript);
    WeatherCommand(std::string transcript, std::optional<WeatherTime> time);
    ~WeatherCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(WeatherCommand)

    bool isComplete() const override;
    std::optional<WeatherTime> time() const;
};

inline std::optional<WeatherTime> WeatherCommand::time() const
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
    std::optional<float> m_volumePercent;

public:
    explicit SetVolumeCommand(std::string transcript);
    SetVolumeCommand(std::string transcript, std::optional<float> volumePercent);
    ~SetVolumeCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(SetVolumeCommand)

    bool isComplete() const override;
    std::optional<float> volumePercent() const;
};

inline std::optional<float> SetVolumeCommand::volumePercent() const
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
    std::optional<AlarmType> m_alarmType;

    std::optional<int> m_weekDay;
    std::optional<Date> m_date;
    std::optional<Time> m_time;

public:
    explicit AddAlarmCommand(std::string transcript);
    AddAlarmCommand(
        std::string transcript,
        std::optional<AlarmType> alarmType,
        std::optional<int> weekDay,
        std::optional<Date> date,
        std::optional<Time> time);
    ~AddAlarmCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddAlarmCommand)

    bool isComplete() const override;

    std::optional<AlarmType> alarmType() const;

    std::optional<int> weekDay() const;
    std::optional<Date> date() const;
    std::optional<Time> time() const;
};

inline std::optional<AlarmType> AddAlarmCommand::alarmType() const
{
    return m_alarmType;
}

inline std::optional<int> AddAlarmCommand::weekDay() const
{
    return m_weekDay;
}

inline std::optional<Date> AddAlarmCommand::date() const
{
    return m_date;
}

inline std::optional<Time> AddAlarmCommand::time() const
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
    std::optional<int64_t> m_id;

public:
    explicit RemoveAlarmCommand(std::string transcript);
    RemoveAlarmCommand(std::string transcript, std::optional<int64_t> id);
    ~RemoveAlarmCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(RemoveAlarmCommand)

    bool isComplete() const override;
    std::optional<int64_t> id() const;
};

inline std::optional<int64_t> RemoveAlarmCommand::id() const
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
    float distance(const std::vector<float>& m_descriptor) const;
    static FaceDescriptor mean(const std::vector<FaceDescriptor>& descriptors);
};

inline const std::vector<float>& FaceDescriptor::data() const
{
    return m_descriptor;
}

class AddReminderCommand : public Command
{
    std::optional<std::string> m_text;
    std::optional<Date> m_date;
    std::optional<Time> m_time;
    std::optional<FaceDescriptor> m_faceDescriptor;

public:
    explicit AddReminderCommand(std::string transcript);
    AddReminderCommand(
        std::string transcript,
        std::optional<std::string> text,
        std::optional<Date> date,
        std::optional<Time> time,
        std::optional<FaceDescriptor> faceDescriptor);
    ~AddReminderCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(AddReminderCommand)

    bool isComplete() const override;
    std::optional<std::string> text() const;
    std::optional<Date> date() const;
    std::optional<Time> time() const;
    std::optional<FaceDescriptor> faceDescriptor() const;
};

inline std::optional<std::string> AddReminderCommand::text() const
{
    return m_text;
}

inline std::optional<Date> AddReminderCommand::date() const
{
    return m_date;
}

inline std::optional<Time> AddReminderCommand::time() const
{
    return m_time;
}

inline std::optional<FaceDescriptor> AddReminderCommand::faceDescriptor() const
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
    std::optional<int64_t> m_id;

public:
    explicit RemoveReminderCommand(std::string transcript);
    RemoveReminderCommand(std::string transcript, std::optional<int64_t> id);
    ~RemoveReminderCommand() override;

    DECLARE_COMMAND_PUBLIC_METHODS(RemoveReminderCommand)

    bool isComplete() const override;
    std::optional<int64_t> id() const;
};

inline std::optional<int64_t> RemoveReminderCommand::id() const
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
