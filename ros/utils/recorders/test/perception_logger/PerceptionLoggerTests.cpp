#include <perception_logger/PerceptionLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(PerceptionLoggerTests, position_aggregateinit_shouldSetAttributes)
{
    Position p{1.0, 2.0, 3.0};
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
    EXPECT_EQ(p.z, 3.0);
}

TEST(PerceptionLoggerTests, imagePosition_aggregateinit_shouldSetAttributes)
{
    ImagePosition p{1.0, 2.0};
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
}

TEST(PerceptionLoggerTests, boundingBox_aggregateinit_shouldSetAttributes)
{
    BoundingBox b{{1.0, 2.0}, 3.0, 4.0};
    EXPECT_EQ(b.centre.x, 1.0);
    EXPECT_EQ(b.centre.y, 2.0);
    EXPECT_EQ(b.width, 3.0);
    EXPECT_EQ(b.height, 4.0);
}

TEST(PerceptionLoggerTests, direction_aggregateinit_shouldSetAttributes)
{
    Direction d{1.0, 2.0, 3.0};
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
