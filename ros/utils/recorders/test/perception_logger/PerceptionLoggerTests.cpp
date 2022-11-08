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

TEST(PerceptionLoggerTests, position_switchEndianness)
{
    Position p = switchEndianness(Position{1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(p.x, 3.0386519416174186e-319);
    EXPECT_DOUBLE_EQ(p.y, 3.1620201333839779e-322);
    EXPECT_DOUBLE_EQ(p.z, 1.0434666440167127e-320);
}

TEST(PerceptionLoggerTests, imagePosition_aggregateinit_shouldSetAttributes)
{
    ImagePosition p{1.0, 2.0};
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
}

TEST(PerceptionLoggerTests, imagePosition_switchEndianness)
{
    ImagePosition p = switchEndianness(ImagePosition{1.0, 2.0});
    EXPECT_DOUBLE_EQ(p.x, 3.0386519416174186e-319);
    EXPECT_DOUBLE_EQ(p.y, 3.1620201333839779e-322);
}

TEST(PerceptionLoggerTests, boundingBox_aggregateinit_shouldSetAttributes)
{
    BoundingBox b{{1.0, 2.0}, 3.0, 4.0};
    EXPECT_EQ(b.center.x, 1.0);
    EXPECT_EQ(b.center.y, 2.0);
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

TEST(PerceptionLoggerTests, direction_switchEndianness)
{
    Direction d = switchEndianness(Direction{1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(d.x, 3.0386519416174186e-319);
    EXPECT_DOUBLE_EQ(d.y, 3.1620201333839779e-322);
    EXPECT_DOUBLE_EQ(d.z, 1.0434666440167127e-320);
}

TEST(PerceptionLoggerTests, timestamp_constructor_shouldSetAttributes)
{
    Timestamp t0(5);
    EXPECT_EQ(t0.unixEpochMs, 5);

    Timestamp t1(ros::Time(5.2));
    EXPECT_EQ(t1.unixEpochMs, 5200);
}
