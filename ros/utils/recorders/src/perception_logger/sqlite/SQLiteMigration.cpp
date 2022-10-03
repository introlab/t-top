#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLiteMigration::SQLiteMigration() {}

SQLiteMigration::SQLiteMigration(const string& sql) : m_sql(sql) {}

SQLiteMigration::~SQLiteMigration() {}

void SQLiteMigration::apply(SQLite::Database& database) const
{
    database.exec(m_sql);
}


static void createVersionTable(SQLite::Database& database)
{
    database.exec("BEGIN;"
                  "CREATE TABLE version(logger TEXT PRIMARY KEY, version INTEGER);"
                  "COMMIT;");
}

static bool hasLoggerVersion(SQLite::Database& database, const string& loggerName)
{
    SQLite::Statement query(database, "SELECT COUNT(*) FROM version WHERE logger=?");
    query.bind(1, loggerName);
    query.executeStep();
    return static_cast<int>(query.getColumn(0)) > 0;
}

static void insertDefaultLoggerVersion(SQLite::Database& database, const string& loggerName)
{
    SQLite::Statement query(database, "INSERT INTO version(logger, version) VALUES(?, 0)");
    query.bind(1, loggerName);
    query.exec();
}

int getLoggerVersion(SQLite::Database& database, const string& loggerName)
{
    SQLite::Statement query(database, "SELECT version FROM version WHERE logger=?");
    query.bind(1, loggerName);
    if (query.executeStep())
    {
        return query.getColumn(0);
    }
    else
    {
        return 0;
    }
}

static int getLoggerVersionOrCreateDefault(SQLite::Database& database, const string& loggerName)
{
    if (!database.tableExists("version"))
    {
        createVersionTable(database);
    }
    if (!hasLoggerVersion(database, loggerName))
    {
        insertDefaultLoggerVersion(database, loggerName);
    }

    return getLoggerVersion(database, loggerName);
}

static void updateLoggerVersion(SQLite::Database& database, const string& loggerName, int version)
{
    SQLite::Statement query(database, "UPDATE version SET version=? WHERE logger=?");
    query.bind(1, version);
    query.bind(2, loggerName);
    query.exec();
}

void applyMigrations(SQLite::Database& database, const string& loggerName, const vector<SQLiteMigration>& migrations)
{
    int loggerVersion = getLoggerVersionOrCreateDefault(database, loggerName);

    for (int i = loggerVersion; i < migrations.size(); i++)
    {
        migrations[i].apply(database);
        updateLoggerVersion(database, loggerName, i + 1);
    }
}
