#include <perception_logger/PerceptionLogger.h>

#include <gtest/gtest.h>

#include "comparisons.h"

using namespace std;

TEST(PerceptionLoggerTests, position_aggregateinit_shouldSetAttributes)
{
    Position p{1.0, 2.0, 3.0};
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
    EXPECT_EQ(p.z, 3.0);
}

TEST(PerceptionLoggerTests, position_toLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 24> BYTES = {byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0xF0}, byte{0x3F},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0},    byte{0x40},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0x08}, byte{0x40}};
    EXPECT_EQ(toLittleEndianBytes(Position{1.0, 2.0, 3.0}), BYTES);
}

TEST(PerceptionLoggerTests, position_fromLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 24> BYTES = {byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0xF0}, byte{0x3F},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0},    byte{0x40},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0x08}, byte{0x40}};
    EXPECT_EQ(fromLittleEndianBytes<Position>(BYTES), (Position{1.0, 2.0, 3.0}));
}


TEST(PerceptionLoggerTests, imagePosition_aggregateinit_shouldSetAttributes)
{
    ImagePosition p{1.0, 2.0};
    EXPECT_EQ(p.x, 1.0);
    EXPECT_EQ(p.y, 2.0);
}

TEST(PerceptionLoggerTests, imagePosition_toLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 16> BYTES = {
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0xF0},
        byte{0x3F},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0x40}};
    EXPECT_EQ(toLittleEndianBytes(ImagePosition{1.0, 2.0}), BYTES);
}

TEST(PerceptionLoggerTests, imagePosition_fromLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 16> BYTES = {
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0xF0},
        byte{0x3F},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0},
        byte{0x40}};
    EXPECT_EQ(fromLittleEndianBytes<ImagePosition>(BYTES), (ImagePosition{1.0, 2.0}));
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

TEST(PerceptionLoggerTests, direction_toLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 24> BYTES = {byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0xF0}, byte{0x3F},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0},    byte{0x40},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0x08}, byte{0x40}};
    EXPECT_EQ(toLittleEndianBytes(Direction{1.0, 2.0, 3.0}), BYTES);
}

TEST(PerceptionLoggerTests, direction_fromLittleEndianBytes_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 24> BYTES = {byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0xF0}, byte{0x3F},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0},    byte{0x40},
                                       byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0}, byte{0x08}, byte{0x40}};
    EXPECT_EQ(fromLittleEndianBytes<Direction>(BYTES), (Direction{1.0, 2.0, 3.0}));
}

TEST(PerceptionLoggerTests, timestamp_constructor_shouldSetAttributes)
{
    Timestamp t0(5);
    EXPECT_EQ(t0.unixEpochMs, 5);

    Timestamp t1(ros::Time(5.2));
    EXPECT_EQ(t1.unixEpochMs, 5200);
}
