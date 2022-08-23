#include <home_logger_common/managers/AlarmManager.h>

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringRessources.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

Alarm::Alarm(Time time) : m_time(move(time)) {}

Alarm::Alarm(int id, Time time) : m_id(id), m_time(move(time)) {}

Alarm::~Alarm() {}

PunctualAlarm::PunctualAlarm(Date date, Time time) : Alarm(move(time)), m_date(move(date)) {}

PunctualAlarm::PunctualAlarm(int id, Date date, Time time) : Alarm(id, move(time)), m_date(move(date)) {}

PunctualAlarm::~PunctualAlarm() {}

string PunctualAlarm::toSpeech()
{
    return Formatter::format(
        StringRessources::getValue("dialogs.commands.alarm.punctual"),
        fmt::arg("id", m_id.value()),
        fmt::arg("date", m_date),
        fmt::arg("time", m_time));
}

void PunctualAlarm::insertAlarm(SQLite::Database& database, int id) const
{
    SQLite::Statement insert(
        database,
        "INSERT INTO alarm(id, type, year, month, day, hour, minute) VALUES(?, ?, ?, ?, ?, ?, ?)");

    insert.bind(1, id);
    insert.bind(2, static_cast<int>(AlarmType::PUNCTUAL));
    insert.bind(3, m_date.year);
    insert.bind(4, m_date.month);
    insert.bind(5, m_date.day);
    insert.bind(6, m_time.hour);
    insert.bind(7, m_time.minute);
    insert.exec();
}

DaylyAlarm::DaylyAlarm(Time time) : Alarm(move(time)) {}

DaylyAlarm::DaylyAlarm(int id, Time time) : Alarm(id, move(time)) {}

DaylyAlarm::~DaylyAlarm() {}

string DaylyAlarm::toSpeech()
{
    return Formatter::format(
        StringRessources::getValue("dialogs.commands.alarm.dayly"),
        fmt::arg("id", m_id.value()),
        fmt::arg("time", m_time));
}

void DaylyAlarm::insertAlarm(SQLite::Database& database, int id) const
{
    SQLite::Statement insert(database, "INSERT INTO alarm(id, type, hour, minute) VALUES(?, ?, ?, ?)");

    insert.bind(1, id);
    insert.bind(2, static_cast<int>(AlarmType::DAYLY));
    insert.bind(3, m_time.hour);
    insert.bind(4, m_time.minute);
    insert.exec();
}

WeeklyAlarm::WeeklyAlarm(int weekDay, Time time) : Alarm(move(time)), m_weekDay(weekDay) {}

WeeklyAlarm::WeeklyAlarm(int id, int weekDay, Time time) : Alarm(id, move(time)), m_weekDay(weekDay) {}

WeeklyAlarm::~WeeklyAlarm() {}

string WeeklyAlarm::toSpeech()
{
    return Formatter::format(
        StringRessources::getValue("dialogs.commands.alarm.weekly"),
        fmt::arg("id", m_id.value()),
        fmt::arg("week_day", Formatter::weekDayNames().at(m_weekDay)),
        fmt::arg("time", m_time));
}

void WeeklyAlarm::insertAlarm(SQLite::Database& database, int id) const
{
    SQLite::Statement insert(database, "INSERT INTO alarm(id, type, week_day, hour, minute) VALUES(?, ?, ?, ?, ?)");

    insert.bind(1, id);
    insert.bind(2, static_cast<int>(AlarmType::WEEKLY));
    insert.bind(3, m_weekDay);
    insert.bind(4, m_time.hour);
    insert.bind(5, m_time.minute);
    insert.exec();
}


unique_ptr<Alarm> toAlarm(const AddAlarmCommand& command)
{
    if (!command.isComplete())
    {
        throw runtime_error("The command must be complete before transforming to an alarm");
    }

    if (command.alarmType() == AlarmType::PUNCTUAL)
    {
        return make_unique<PunctualAlarm>(command.date().value(), command.time().value());
    }

    if (command.alarmType() == AlarmType::DAYLY)
    {
        return make_unique<DaylyAlarm>(command.time().value());
    }

    return make_unique<WeeklyAlarm>(command.weekDay().value(), command.time().value());
}


AlarmManager::AlarmManager(SQLite::Database& database) : m_database(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE alarm("
                                                       "    id INTEGER PRIMARY KEY,"
                                                       "    type INTEGER,"
                                                       "    week_day INTEGER,"
                                                       "    year INTEGER,"
                                                       "    month INTEGER,"
                                                       "    day INTEGER,"
                                                       "    hour INTEGER,"
                                                       "    minute INTEGER"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "alarm", migrations);
}

AlarmManager::~AlarmManager() {}

void AlarmManager::insertAlarm(unique_ptr<Alarm> alarm)
{
    alarm->insertAlarm(m_database, getNextId());
}

void AlarmManager::removeAlarm(int id)
{
    SQLite::Statement deleteFrom(m_database, "DELETE FROM alarm WHERE id=?");

    deleteFrom.bind(1, id);
    deleteFrom.exec();
}

vector<unique_ptr<Alarm>> AlarmManager::listAlarms()
{
    vector<unique_ptr<Alarm>> alarms;

    SQLite::Statement query(m_database, "SELECT id, type, week_day, year, month, day, hour, minute FROM alarm ORDER BY id");

    while (query.executeStep())
    {
        alarms.emplace_back(alarmFromRow(query));
    }

    return alarms;
}

int AlarmManager::getNextId()
{
    SQLite::Statement query(m_database, "SELECT MIN(t1.id) FROM "
                                        "("
                                        "    SELECT 1 AS id "
                                        "    UNION ALL "
                                        "    SELECT id + 1 from alarm "
                                        ") as t1 "
                                        "LEFT OUTER JOIN alarm as t2 ON t1.id = t2.id WHERE t2.id IS NULL");
    if (query.executeStep())
    {
        return query.getColumn(0);
    }
    throw runtime_error("getNextId failed");
}

unique_ptr<Alarm> AlarmManager::alarmFromRow(SQLite::Statement& query)
{
    int id = query.getColumn(0);
    Time time(query.getColumn(6), query.getColumn(7));

    switch (static_cast<AlarmType>(static_cast<int>(query.getColumn(1))))
    {
        case AlarmType::PUNCTUAL:
            return make_unique<PunctualAlarm>(
                id,
                Date(query.getColumn(3), query.getColumn(4), query.getColumn(5)),
                time);

        case AlarmType::DAYLY:
            return make_unique<DaylyAlarm>(id, time);

        case AlarmType::WEEKLY:
            return make_unique<WeeklyAlarm>(id, query.getColumn(2), time);
    }

    throw runtime_error("Invalid alarm type in the database");
}
