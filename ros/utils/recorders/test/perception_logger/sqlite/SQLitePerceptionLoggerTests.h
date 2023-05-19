#ifndef SQLITE_PERCEPTION_LOGGER_TESTS_H
#define SQLITE_PERCEPTION_LOGGER_TESTS_H

#include <perception_logger/PerceptionLogger.h>
#include <perception_logger/BinarySerialization.h>

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
        const std::byte* data = reinterpret_cast<const std::byte*>(column.getBlob());
        size_t size = column.size() / sizeof(T);
        vec.clear();

        for (size_t i = 0; i < size; i++)
        {
            std::array<std::byte, sizeof(T)> bytes;
            memcpy(bytes.data(), data + i * sizeof(T), sizeof(T));
            vec.push_back(fromLittleEndianBytes<T>(bytes));
        }
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
