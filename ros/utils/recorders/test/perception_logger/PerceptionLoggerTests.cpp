#include <perception_logger/PerceptionLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(PerceptionLoggerTests, position_constructor_shouldSetAttributes)
{
    Position p(1.0, 2.0, 3.0);
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
    EXPECT_EQ(p.z, 3.0);
}

TEST(PerceptionLoggerTests, direction_constructor_shouldSetAttributes)
{
    Direction d(1.0, 2.0, 3.0);
    EXPECT_EQ(d.x, 1.0);
    EXPECT_EQ(d.y, 2.0);
    EXPECT_EQ(d.z, 3.0);
}

TEST(PerceptionLoggerTests, timestamp_constructor_shouldSetAttributes)
{
    Timestamp t0(5);
    EXPECT_EQ(t0.unixEpoch, 5);

    Timestamp t1(ros::Time(5.2));
    EXPECT_EQ(t1.unixEpoch, 5);
}
