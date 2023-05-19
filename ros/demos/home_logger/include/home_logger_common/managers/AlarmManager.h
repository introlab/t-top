#ifndef HOME_LOGGER_COMMON_MANAGERS_ALARM_MANAGER_H
#define HOME_LOGGER_COMMON_MANAGERS_ALARM_MANAGER_H

#include <home_logger_common/DateTime.h>
#include <home_logger_common/commands/Commands.h>

#include <SQLiteCpp/SQLiteCpp.h>

#include <hbba_lite/utils/ClassMacros.h>

#include <memory>
#include <string>
#include <vector>

class AlarmManager;

class Alarm
{
protected:
    std::optional<int> m_id;
    Time m_time;

public:
    explicit Alarm(Time time);
    Alarm(int id, Time time);
    virtual ~Alarm();

    std::optional<int> id() const;
    Time time() const;

    virtual std::string toSpeech() = 0;

protected:
    virtual void insertAlarm(SQLite::Database& database, int id) const = 0;

    friend AlarmManager;
};

inline std::optional<int> Alarm::id() const
{
    return m_id;
}

inline Time Alarm::time() const
{
    return m_time;
}

class PunctualAlarm : public Alarm
{
    Date m_date;

public:
    PunctualAlarm(Date date, Time time);
    PunctualAlarm(int id, Date date, Time time);
    ~PunctualAlarm() override;

    Date date() const;

    std::string toSpeech() override;

protected:
    void insertAlarm(SQLite::Database& database, int id) const override;

    static std::vector<std::unique_ptr<Alarm>> listDueAlarms(SQLite::Database& database, DateTime now);

    friend AlarmManager;
};

inline Date PunctualAlarm::date() const
{
    return m_date;
}

class DailyAlarm : public Alarm
{
public:
    explicit DailyAlarm(Time time);
    DailyAlarm(int id, Time time);
    ~DailyAlarm() override;

    std::string toSpeech() override;

protected:
    void insertAlarm(SQLite::Database& database, int id) const override;

    static std::vector<std::unique_ptr<Alarm>> listDueAlarms(SQLite::Database& database, DateTime now);

    friend AlarmManager;
};

class WeeklyAlarm : public Alarm
{
    int m_weekDay;

public:
    WeeklyAlarm(int weekDay, Time time);
    WeeklyAlarm(int id, int weekDay, Time time);
    ~WeeklyAlarm() override;

    int weekDay();

    std::string toSpeech() override;

protected:
    void insertAlarm(SQLite::Database& database, int id) const override;

    static std::vector<std::unique_ptr<Alarm>> listDueAlarms(SQLite::Database& database, DateTime now);

    friend AlarmManager;
};

inline int WeeklyAlarm::weekDay()
{
    return m_weekDay;
}


std::unique_ptr<Alarm> toAlarm(const AddAlarmCommand& command);


class AlarmManager
{
    SQLite::Database& m_database;

public:
    explicit AlarmManager(SQLite::Database& database);
    virtual ~AlarmManager();

    DECLARE_NOT_COPYABLE(AlarmManager);
    DECLARE_NOT_MOVABLE(AlarmManager);

    void insertAlarm(std::unique_ptr<Alarm> alarm);
    void removeAlarm(int id);
    std::vector<std::unique_ptr<Alarm>> listAlarms();

    std::vector<std::unique_ptr<Alarm>> listDueAlarms(DateTime now);
    void informPerformedAlarms(const std::vector<int> alarmIds);

private:
    int getNextId();
};

#endif
