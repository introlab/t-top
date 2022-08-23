#ifndef HOME_LOGGER_COMMON_MANAGERS_ALARM_MANAGER_H
#define HOME_LOGGER_COMMON_MANAGERS_ALARM_MANAGER_H

#include <home_logger_common/DateTime.h>
#include <home_logger_common/commands/Commands.h>

#include <SQLiteCpp/SQLiteCpp.h>

#include <memory>
#include <string>
#include <vector>

class AlarmManager;

class Alarm
{
protected:
    tl::optional<int> m_id;
    Time m_time;

public:
    Alarm(Time time);
    Alarm(int id, Time time);
    virtual ~Alarm();

    tl::optional<int> id() const;
    Time time() const;

    virtual std::string toSpeech() = 0;

protected:
    virtual void insertAlarm(SQLite::Database& database) const = 0;

    friend AlarmManager;
};

inline tl::optional<int> Alarm::id() const
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
    void insertAlarm(SQLite::Database& database) const override;
};

inline Date PunctualAlarm::date() const
{
    return m_date;
}

class DaylyAlarm : public Alarm
{
public:
    DaylyAlarm(Time time);
    DaylyAlarm(int id, Time time);
    ~DaylyAlarm() override;

    std::string toSpeech() override;

protected:
    void insertAlarm(SQLite::Database& database) const override;
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
    void insertAlarm(SQLite::Database& database) const override;
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
    AlarmManager(SQLite::Database& database);
    virtual ~AlarmManager();

    void insertAlarm(std::unique_ptr<Alarm> alarm);
    void removeAlarm(int id);
    std::vector<std::unique_ptr<Alarm>> listAlarms();

private:
    std::unique_ptr<Alarm> alarmFromRow(SQLite::Statement& query);
};

#endif
