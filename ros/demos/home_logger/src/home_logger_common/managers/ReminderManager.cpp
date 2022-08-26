#include <home_logger_common/managers/ReminderManager.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

Reminder::Reminder(string text, DateTime datetime, FaceDescriptor faceDescriptor)
    : m_text(move(text)),
      m_datetime(datetime),
      m_faceDescriptor(move(faceDescriptor))
{
}

Reminder::Reminder(int id, string text, DateTime datetime, FaceDescriptor faceDescriptor)
    : m_id(id),
      m_text(move(text)),
      m_datetime(datetime),
      m_faceDescriptor(move(faceDescriptor))
{
}

Reminder::~Reminder() {}


ReminderManager::ReminderManager(SQLite::Database& database) : m_database(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE reminder("
                                                       "    id INTEGER PRIMARY KEY,"
                                                       "    text TEXT,"
                                                       "    face_descriptor BLOB,"
                                                       "    year INTEGER,"
                                                       "    month INTEGER,"
                                                       "    day INTEGER,"
                                                       "    hour INTEGER,"
                                                       "    minute INTEGER"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "reminder", migrations);
}

ReminderManager::~ReminderManager() {}

void ReminderManager::insertReminder(const Reminder& reminder)
{
    SQLite::Statement insert(
        m_database,
        "INSERT INTO reminder(id, text, face_descriptor, year, month, day, hour, minute) VALUES(?, ?, ?, ?, ?, ?, ?, "
        "?)");

    insert.bind(1, getNextId());
    insert.bindNoCopy(2, reminder.text());
    if (reminder.faceDescriptor().data().size() == 0)
    {
        insert.bind(3);
    }
    else
    {
        insert.bindNoCopy(
            3,
            reminder.faceDescriptor().data().data(),
            sizeof(float) * reminder.faceDescriptor().data().size());
    }
    insert.bind(4, reminder.datetime().date.year);
    insert.bind(5, reminder.datetime().date.month);
    insert.bind(6, reminder.datetime().date.day);
    insert.bind(7, reminder.datetime().time.hour);
    insert.bind(8, reminder.datetime().time.minute);
    insert.exec();
}

void ReminderManager::removeReminder(int id)
{
    SQLite::Statement deleteFrom(m_database, "DELETE FROM reminder WHERE id=?");

    deleteFrom.bind(1, id);
    deleteFrom.exec();
}

void ReminderManager::removeRemindersOlderThan(DateTime datetime)
{
    SQLite::Statement deleteFrom(
        m_database,
        "DELETE FROM reminder WHERE "
        "year<?1 OR "
        "(year=?1 AND month<?2) OR"
        "(year=?1 AND month=?2 AND day<?3) OR"
        "(year=?1 AND month=?2 AND day=?3 AND hour<?4) OR"
        "(year=?1 AND month=?2 AND day=?3 AND hour=?4 AND minute<?5)");

    deleteFrom.bind(1, datetime.date.year);
    deleteFrom.bind(2, datetime.date.month);
    deleteFrom.bind(3, datetime.date.day);
    deleteFrom.bind(4, datetime.time.hour);
    deleteFrom.bind(5, datetime.time.minute);
    deleteFrom.exec();
}

vector<Reminder> ReminderManager::listReminders()
{
    vector<Reminder> reminders;

    SQLite::Statement query(
        m_database,
        "SELECT id, text, face_descriptor, year, month, day, hour, minute FROM reminder ORDER BY id");

    while (query.executeStep())
    {
        reminders.emplace_back(reminderFromRow(query));
    }

    return reminders;
}

vector<Reminder> ReminderManager::listReminders(const Date& date)
{
    vector<Reminder> reminders;

    SQLite::Statement query(
        m_database,
        "SELECT id, text, face_descriptor, year, month, day, hour, minute FROM reminder "
        "WHERE year=? AND month=? AND day=? ORDER BY id");
    query.bind(1, date.year);
    query.bind(2, date.month);
    query.bind(3, date.day);

    while (query.executeStep())
    {
        reminders.emplace_back(reminderFromRow(query));
    }

    return reminders;
}

int ReminderManager::getNextId()
{
    SQLite::Statement query(
        m_database,
        "SELECT MIN(t1.id) FROM "
        "("
        "    SELECT 1 AS id "
        "    UNION ALL "
        "    SELECT id + 1 from reminder "
        ") as t1 "
        "LEFT OUTER JOIN reminder as t2 ON t1.id = t2.id WHERE t2.id IS NULL");
    if (query.executeStep())
    {
        return query.getColumn(0);
    }
    throw runtime_error("getNextId failed");
}

Reminder ReminderManager::reminderFromRow(SQLite::Statement& query)
{
    DateTime datetime(
        Date(query.getColumn(3), query.getColumn(4), query.getColumn(5)),
        Time(query.getColumn(6), query.getColumn(7)));
    auto faceDescriptorColumn = query.getColumn(2);

    FaceDescriptor faceDescriptor(vector<float>(
        reinterpret_cast<const float*>(faceDescriptorColumn.getBlob()),
        reinterpret_cast<const float*>(faceDescriptorColumn.getBlob()) + faceDescriptorColumn.size() / sizeof(float)));

    return Reminder(query.getColumn(0), query.getColumn(1), datetime, faceDescriptor);
}
