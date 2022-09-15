#include <home_logger_common/managers/AlarmManager.h>

#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

#include <algorithm>
#include <sstream>

using namespace std;

static unique_ptr<Alarm> alarmFromRow(SQLite::Statement& query)
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

        case AlarmType::DAILY:
            return make_unique<DailyAlarm>(id, time);

        case AlarmType::WEEKLY:
            return make_unique<WeeklyAlarm>(id, query.getColumn(2), time);
    }

    throw runtime_error("Invalid alarm type in the database");
}

Alarm::Alarm(Time time) : m_time(move(time)) {}

Alarm::Alarm(int id, Time time) : m_id(id), m_time(move(time)) {}

Alarm::~Alarm() {}

PunctualAlarm::PunctualAlarm(Date date, Time time) : Alarm(move(time)), m_date(move(date)) {}

PunctualAlarm::PunctualAlarm(int id, Date date, Time time) : Alarm(id, move(time)), m_date(move(date)) {}

PunctualAlarm::~PunctualAlarm() {}

string PunctualAlarm::toSpeech()
{
    return Formatter::format(
        StringResources::getValue("dialogs.commands.alarm.punctual"),
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

vector<unique_ptr<Alarm>> PunctualAlarm::listDueAlarms(SQLite::Database& database, DateTime now)
{
    vector<unique_ptr<Alarm>> alarms;

    SQLite::Statement query(
        database,
        "SELECT id, type, week_day, year, month, day, hour, minute, last_alarm_unixepoch FROM alarm "
        "WHERE type=? AND year=? AND month=? AND day=? AND hour=? AND minute<=? AND "
        "    (STRFTIME('%s') - last_alarm_unixepoch) > 3600 "
        "ORDER BY id");

    query.bind(1, static_cast<int>(AlarmType::PUNCTUAL));
    query.bind(2, now.date.year);
    query.bind(3, now.date.month);
    query.bind(4, now.date.day);
    query.bind(5, now.time.hour);
    query.bind(6, now.time.minute);

    while (query.executeStep())
    {
        alarms.emplace_back(alarmFromRow(query));
    }

    return alarms;
}

DailyAlarm::DailyAlarm(Time time) : Alarm(move(time)) {}

DailyAlarm::DailyAlarm(int id, Time time) : Alarm(id, move(time)) {}

DailyAlarm::~DailyAlarm() {}

string DailyAlarm::toSpeech()
{
    return Formatter::format(
        StringResources::getValue("dialogs.commands.alarm.daily"),
        fmt::arg("id", m_id.value()),
        fmt::arg("time", m_time));
}

void DailyAlarm::insertAlarm(SQLite::Database& database, int id) const
{
    SQLite::Statement insert(database, "INSERT INTO alarm(id, type, hour, minute) VALUES(?, ?, ?, ?)");

    insert.bind(1, id);
    insert.bind(2, static_cast<int>(AlarmType::DAILY));
    insert.bind(3, m_time.hour);
    insert.bind(4, m_time.minute);
    insert.exec();
}

vector<unique_ptr<Alarm>> DailyAlarm::listDueAlarms(SQLite::Database& database, DateTime now)
{
    vector<unique_ptr<Alarm>> alarms;

    SQLite::Statement query(
        database,
        "SELECT id, type, week_day, year, month, day, hour, minute, last_alarm_unixepoch FROM alarm "
        "WHERE type=? AND hour=? AND minute<=? AND (STRFTIME('%s') - last_alarm_unixepoch) > 3600 "
        "ORDER BY id");

    query.bind(1, static_cast<int>(AlarmType::DAILY));
    query.bind(2, now.time.hour);
    query.bind(3, now.time.minute);

    while (query.executeStep())
    {
        alarms.emplace_back(alarmFromRow(query));
    }

    return alarms;
}

WeeklyAlarm::WeeklyAlarm(int weekDay, Time time) : Alarm(move(time)), m_weekDay(weekDay) {}

WeeklyAlarm::WeeklyAlarm(int id, int weekDay, Time time) : Alarm(id, move(time)), m_weekDay(weekDay) {}

WeeklyAlarm::~WeeklyAlarm() {}

string WeeklyAlarm::toSpeech()
{
    return Formatter::format(
        StringResources::getValue("dialogs.commands.alarm.weekly"),
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

vector<unique_ptr<Alarm>> WeeklyAlarm::listDueAlarms(SQLite::Database& database, DateTime now)
{
    vector<unique_ptr<Alarm>> alarms;

    SQLite::Statement query(
        database,
        "SELECT id, type, week_day, year, month, day, hour, minute, last_alarm_unixepoch FROM alarm "
        "WHERE type=? AND week_day=? AND hour=? AND minute<=? AND (STRFTIME('%s') - last_alarm_unixepoch) > 3600 "
        "ORDER BY id");

    query.bind(1, static_cast<int>(AlarmType::WEEKLY));
    query.bind(2, now.date.weekDay());
    query.bind(3, now.time.hour);
    query.bind(4, now.time.minute);

    while (query.executeStep())
    {
        alarms.emplace_back(alarmFromRow(query));
    }

    return alarms;
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

    if (command.alarmType() == AlarmType::DAILY)
    {
        return make_unique<DailyAlarm>(command.time().value());
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
                                                       "    minute INTEGER,"
                                                       "    last_alarm_unixepoch INTEGER DEFAULT 0"
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

    SQLite::Statement query(
        m_database,
        "SELECT id, type, week_day, year, month, day, hour, minute FROM alarm ORDER BY id");

    while (query.executeStep())
    {
        alarms.emplace_back(alarmFromRow(query));
    }

    return alarms;
}

vector<unique_ptr<Alarm>> AlarmManager::listDueAlarms(DateTime now)
{
    vector<unique_ptr<Alarm>> alarms;
    auto punctualAlarms = PunctualAlarm::listDueAlarms(m_database, now);
    auto dailyAlarms = DailyAlarm::listDueAlarms(m_database, now);
    auto weeklyAlarms = WeeklyAlarm::listDueAlarms(m_database, now);

    alarms.insert(alarms.end(), make_move_iterator(punctualAlarms.begin()), make_move_iterator(punctualAlarms.end()));
    alarms.insert(alarms.end(), make_move_iterator(dailyAlarms.begin()), make_move_iterator(dailyAlarms.end()));
    alarms.insert(alarms.end(), make_move_iterator(weeklyAlarms.begin()), make_move_iterator(weeklyAlarms.end()));

    return alarms;
}

void AlarmManager::informPerformedAlarms(const vector<int> alarmIds)
{
    if (alarmIds.empty())
    {
        return;
    }

    stringstream inClause;
    for (int i = 0; i < alarmIds.size(); i++)
    {
        if (i > 0)
        {
            inClause << ", ";
        }
        inClause << "?";
    }

    SQLite::Statement deleteFrom(m_database, "DELETE FROM alarm WHERE type=? AND id IN (" + inClause.str() + ")");
    deleteFrom.bind(1, static_cast<int>(AlarmType::PUNCTUAL));

    SQLite::Statement update(
        m_database,
        "UPDATE alarm SET last_alarm_unixepoch=STRFTIME('%s') WHERE id IN (" + inClause.str() + ")");


    for (int i = 0; i < alarmIds.size(); i++)
    {
        deleteFrom.bind(i + 2, alarmIds[i]);
        update.bind(i + 1, alarmIds[i]);
    }

    deleteFrom.exec();
    update.exec();
}

int AlarmManager::getNextId()
{
    SQLite::Statement query(
        m_database,
        "SELECT MIN(t1.id) FROM "
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
