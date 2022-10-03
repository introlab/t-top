#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_MIGRATION_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_MIGRATION_H

#include <SQLiteCpp/SQLiteCpp.h>

#include <vector>
#include <memory>
#include <string>

class SQLiteMigration
{
    std::string m_sql;

public:
    SQLiteMigration();
    SQLiteMigration(const std::string& sql);
    virtual ~SQLiteMigration();

    void apply(SQLite::Database& database) const;
};

int getLoggerVersion(SQLite::Database& database, const std::string& loggerName);
void applyMigrations(
    SQLite::Database& database,
    const std::string& loggerName,
    const std::vector<SQLiteMigration>& migrations);

#endif
