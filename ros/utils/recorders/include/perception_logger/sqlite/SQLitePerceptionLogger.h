#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_PERCEPTION_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_PERCEPTION_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <SQLiteCpp/SQLiteCpp.h>

#include <tl/optional.hpp>

#include <vector>
#include <memory>
#include <string>

class SQLitePerceptionLogger
{
protected:
    SQLite::Database& m_database;

public:
    SQLitePerceptionLogger(SQLite::Database& database);
    virtual ~SQLitePerceptionLogger();

    int64_t insertPerception(Timestamp timestamp, tl::optional<Position> position, tl::optional<Direction> direction);
};


#endif
