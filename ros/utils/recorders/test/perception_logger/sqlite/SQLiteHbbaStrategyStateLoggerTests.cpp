#include <perception_logger/sqlite/SQLiteHbbaStrategyStateLogger.h>

#include <gtest/gtest.h>

using namespace std;

void readHbbaStrategyState(
    SQLite::Database& database,
    int64_t id,
    int64_t& timestampMs,
    string& desireTypeName,
    string& strategyTypeName,
    bool& enabled)
{
    SQLite::Statement query(
        database,
        "SELECT timestamp_ms, desire_type_name, strategy_type_name, enabled"
        "    FROM hbba_strategy_state WHERE id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        timestampMs = -1;
        desireTypeName = "";
        strategyTypeName = "";
        enabled = false;
        return;
    }

    timestampMs = query.getColumn(0).getInt64();
    desireTypeName = query.getColumn(1).getString();
    strategyTypeName = query.getColumn(2).getString();
    enabled = static_cast<bool>(query.getColumn(3).getInt());
}

TEST(SQLiteHbbaStrategyStateLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteHbbaStrategyStateLogger logger(database);

    int64_t id0 = logger.log(HbbaStrategyState(Timestamp(101), "d1", "s1", true));
    int64_t id1 = logger.log(HbbaStrategyState(Timestamp(102), "d2", "s2", false));

    int64_t timestampMs;
    string desireTypeName;
    string strategyTypeName;
    bool enabled;

    readHbbaStrategyState(database, id0, timestampMs, desireTypeName, strategyTypeName, enabled);
    EXPECT_EQ(timestampMs, 101);
    EXPECT_EQ(desireTypeName, "d1");
    EXPECT_EQ(strategyTypeName, "s1");
    EXPECT_TRUE(enabled);

    readHbbaStrategyState(database, id1, timestampMs, desireTypeName, strategyTypeName, enabled);
    EXPECT_EQ(timestampMs, 102);
    EXPECT_EQ(desireTypeName, "d2");
    EXPECT_EQ(strategyTypeName, "s2");
    EXPECT_FALSE(enabled);
}
