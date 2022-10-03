#include <perception_logger/VideoAnalysisLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(VideoAnalysisLoggerTests, videoAnalysis_constructor_shouldSetAttributes)
{
    VideoAnalysis analysis0(Timestamp(1), Position(2.0, 3.0, 4.0), Direction(5.0, 6.0, 7.0), "a");
    EXPECT_EQ(analysis0.timestamp.unixEpoch, 1);
    EXPECT_EQ(analysis0.position.x, 2.0);
    EXPECT_EQ(analysis0.position.y, 3.0);
    EXPECT_EQ(analysis0.position.z, 4.0);
    EXPECT_EQ(analysis0.direction.x, 5.0);
    EXPECT_EQ(analysis0.direction.y, 6.0);
    EXPECT_EQ(analysis0.direction.z, 7.0);
    EXPECT_EQ(analysis0.objectClass, "a");
    EXPECT_EQ(analysis0.personPose, tl::nullopt);
    EXPECT_EQ(analysis0.personPoseConfidence, tl::nullopt);
    EXPECT_EQ(analysis0.faceDescriptor, tl::nullopt);

    VideoAnalysis analysis1(
        Timestamp(10),
        Position(20.0, 30.0, 40.0),
        Direction(50.0, 60.0, 70.0),
        "b",
        {Position(80.0, 90.0, 100.0)},
        {0.5f});
    EXPECT_EQ(analysis1.timestamp.unixEpoch, 10);
    EXPECT_EQ(analysis1.position.x, 20.0);
    EXPECT_EQ(analysis1.position.y, 30.0);
    EXPECT_EQ(analysis1.position.z, 40.0);
    EXPECT_EQ(analysis1.direction.x, 50.0);
    EXPECT_EQ(analysis1.direction.y, 60.0);
    EXPECT_EQ(analysis1.direction.z, 70.0);
    EXPECT_EQ(analysis1.objectClass, "b");
    EXPECT_EQ(analysis1.personPose, vector<Position>({Position(80.0, 90.0, 100.0)}));
    EXPECT_EQ(analysis1.personPoseConfidence, vector<float>({0.5}));
    EXPECT_EQ(analysis1.faceDescriptor, tl::nullopt);

    VideoAnalysis analysis2(
        Timestamp(10),
        Position(20.0, 30.0, 40.0),
        Direction(50.0, 60.0, 70.0),
        "b",
        {Position(80.0, 90.0, 100.0)},
        {0.5f},
        {200.f});
    EXPECT_EQ(analysis2.timestamp.unixEpoch, 10);
    EXPECT_EQ(analysis2.position.x, 20.0);
    EXPECT_EQ(analysis2.position.y, 30.0);
    EXPECT_EQ(analysis2.position.z, 40.0);
    EXPECT_EQ(analysis2.direction.x, 50.0);
    EXPECT_EQ(analysis2.direction.y, 60.0);
    EXPECT_EQ(analysis2.direction.z, 70.0);
    EXPECT_EQ(analysis2.objectClass, "b");
    EXPECT_EQ(analysis2.personPose, vector<Position>({Position(80.0, 90.0, 100.0)}));
    EXPECT_EQ(analysis2.personPoseConfidence, vector<float>({0.5f}));
    EXPECT_EQ(analysis2.faceDescriptor, vector<float>({200.f}));
}
