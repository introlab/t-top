#ifndef SQLITE_PERCEPTION_LOGGER_TESTS_H
#define SQLITE_PERCEPTION_LOGGER_TESTS_H

#include <perception_logger/PerceptionLogger.h>

#include <SQLiteCpp/SQLiteCpp.h>

#include <vector>

template<class T>
void columnToVector(const SQLite::Column& column, std::vector<T>& vec)
{
    if (column.isNull())
    {
        vec.clear();
    }
    else
    {
        const T* data = reinterpret_cast<const T*>(column.getBlob());
        size_t size = column.size() / sizeof(T);
        vec = std::vector<T>(data, data + size);
    }
}

bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp);
bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp, Position position);
bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp, Direction direction);
bool perceptionExists(
    SQLite::Database& database,
    int64_t id,
    Timestamp timestamp,
    Position position,
    Direction direction);

#endif
