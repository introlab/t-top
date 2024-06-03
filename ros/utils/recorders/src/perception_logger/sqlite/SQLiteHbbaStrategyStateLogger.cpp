#include <perception_logger/sqlite/SQLiteHbbaStrategyStateLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLiteHbbaStrategyStateLogger::SQLiteHbbaStrategyStateLogger(SQLite::Database& database) : m_database(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE hbba_strategy_state("
                                                       "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                                                       "    timestamp_ms INTEGER,"
                                                       "    desire_type_name TEXT,"
                                                       "    strategy_type_name TEXT,"
                                                       "    enabled INTEGER"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "hbba_strategy_state", migrations);
}

SQLiteHbbaStrategyStateLogger::~SQLiteHbbaStrategyStateLogger() {}

int64_t SQLiteHbbaStrategyStateLogger::log(const HbbaStrategyState& state)
{
    SQLite::Statement insert(
        m_database,
        "INSERT INTO hbba_strategy_state(timestamp_ms, desire_type_name, strategy_type_name, enabled)"
        "    VALUES(?, ?, ?, ?)");
    insert.clearBindings();
    insert.bind(1, state.timestamp.unixEpochMs);
    insert.bindNoCopy(2, state.desireTypeName);
    insert.bindNoCopy(3, state.strategyTypeName);
    insert.bind(4, static_cast<int>(state.enabled));

    insert.exec();
    return m_database.getLastInsertRowid();
}
