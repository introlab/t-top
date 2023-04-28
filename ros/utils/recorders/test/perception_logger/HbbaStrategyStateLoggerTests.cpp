#include <perception_logger/HbbaStrategyStateLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(HbbaStrategyStateLoggerTests, hbbaStrategyState_constructor_shouldSetAttributes)
{
    HbbaStrategyState state(Timestamp(1), "d", "s", true);
    EXPECT_EQ(state.timestamp.unixEpochMs, 1);
    EXPECT_EQ(state.desireTypeName, "d");
    EXPECT_EQ(state.strategyTypeName, "s");
    EXPECT_TRUE(state.enabled);
}
