#include <perception_logger/sqlite/SQLitePerceptionLogger.h>

#include <gtest/gtest.h>

using namespace std;

bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp)
{
    SQLite::Statement query(
        database,
        "SELECT COUNT(*) FROM perception "
        "WHERE id=? AND timestamp_ms=? AND "
        "      position_x is NULL AND position_y is NULL AND position_z is NULL AND "
        "      direction_x is NULL AND direction_y is NULL AND direction_z is NULL");

    query.bind(1, id);
    query.bind(2, timestamp.unixEpochMs);

    query.executeStep();
    return static_cast<int>(query.getColumn(0)) != 0;
}

bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp, Position position)
{
    SQLite::Statement query(
        database,
        "SELECT COUNT(*) FROM perception "
        "WHERE id=? AND timestamp_ms=? AND "
        "      position_x=? AND position_y=? AND position_z=? AND "
        "      direction_x is NULL AND direction_y is NULL AND direction_z is NULL");

    query.bind(1, id);
    query.bind(2, timestamp.unixEpochMs);
    query.bind(3, position.x);
    query.bind(4, position.y);
    query.bind(5, position.z);
    query.executeStep();
    return static_cast<int>(query.getColumn(0)) != 0;
}

bool perceptionExists(SQLite::Database& database, int64_t id, Timestamp timestamp, Direction direction)
{
    SQLite::Statement query(
        database,
        "SELECT COUNT(*) FROM perception "
        "WHERE id=? AND timestamp_ms=? AND "
        "      position_x is NULL AND position_y is NULL AND position_z is NULL AND "
        "      direction_x=? AND direction_y=? AND direction_z=?");

    query.bind(1, id);
    query.bind(2, timestamp.unixEpochMs);
    query.bind(3, direction.x);
    query.bind(4, direction.y);
    query.bind(5, direction.z);
    query.executeStep();
    return static_cast<int>(query.getColumn(0)) != 0;
}

bool perceptionExists(
    SQLite::Database& database,
    int64_t id,
    Timestamp timestamp,
    Position position,
    Direction direction)
{
    SQLite::Statement query(
        database,
        "SELECT COUNT(*) FROM perception "
        "WHERE id=? AND timestamp_ms=? AND "
        "      position_x=? AND position_y=? AND position_z=? AND "
        "      direction_x=? AND direction_y=? AND direction_z=?");

    query.bind(1, id);
    query.bind(2, timestamp.unixEpochMs);
    query.bind(3, position.x);
    query.bind(4, position.y);
    query.bind(5, position.z);
    query.bind(6, direction.x);
    query.bind(7, direction.y);
    query.bind(8, direction.z);
    query.executeStep();
    return static_cast<int>(query.getColumn(0)) != 0;
}


TEST(SQLitePerceptionLoggerTests, insertPerception_shouldInsertAndReturnId)
{
    const string loggerName = "audio";
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLitePerceptionLogger logger(database);

    int64_t id0 = logger.insertPerception(Timestamp(101), nullopt, nullopt);
    int64_t id1 = logger.insertPerception(Timestamp(102), Position{10, 11, 12}, nullopt);
    int64_t id2 = logger.insertPerception(Timestamp(103), nullopt, Direction{20, 21, 22});
    int64_t id3 = logger.insertPerception(Timestamp(104), Position{30, 31, 32}, Direction{33, 34, 35});

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101)));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102), Position{10, 11, 12}));
    EXPECT_TRUE(perceptionExists(database, id2, Timestamp(103), Direction{20, 21, 22}));
    EXPECT_TRUE(perceptionExists(database, id3, Timestamp(104), Position{30, 31, 32}, Direction{33, 34, 35}));
}
