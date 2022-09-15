#include <perception_logger/sqlite/SQLitePerceptionLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLitePerceptionLogger::SQLitePerceptionLogger(SQLite::Database& database) : m_database(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE perception("
                                                       "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                                                       "    timestamp INTEGER,"
                                                       "    position_x REAL,"
                                                       "    position_y REAL,"
                                                       "    position_z REAL,"
                                                       "    direction_x REAL,"
                                                       "    direction_y REAL,"
                                                       "    direction_z REAL"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "perception", migrations);
}

SQLitePerceptionLogger::~SQLitePerceptionLogger() {}

int64_t SQLitePerceptionLogger::insertPerception(
    Timestamp timestamp,
    tl::optional<Position> position,
    tl::optional<Direction> direction)
{
    SQLite::Statement insert(
        m_database,
        "INSERT INTO perception(timestamp, position_x, position_y, position_z, direction_x, direction_y, direction_z) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)");
    insert.clearBindings();
    insert.bind(1, timestamp.unixEpoch);

    if (position.has_value())
    {
        insert.bind(2, position.value().x);
        insert.bind(3, position.value().y);
        insert.bind(4, position.value().z);
    }
    if (direction.has_value())
    {
        insert.bind(5, direction.value().x);
        insert.bind(6, direction.value().y);
        insert.bind(7, direction.value().z);
    }

    insert.exec();
    return m_database.getLastInsertRowid();
}
