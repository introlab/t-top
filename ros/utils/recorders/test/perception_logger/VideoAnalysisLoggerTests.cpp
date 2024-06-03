#include <perception_logger/VideoAnalysisLogger.h>

#include <gtest/gtest.h>

using namespace std;

#include "comparisons.h"

TEST(VideoAnalysisLoggerTests, videoAnalysis_constructor_shouldSetAttributes)
{
    VideoAnalysis analysis0{
        Timestamp(1),
        Position{2.0, 3.0, 4.0},
        Direction{5.0, 6.0, 7.0},
        "a",
        0.75f,
        0.5f,
        BoundingBox{{8.0, 9.0}, 10.0, 11.0}};
    EXPECT_EQ(analysis0.timestamp.unixEpochMs, 1);
    EXPECT_EQ(analysis0.position.x, 2.0);
    EXPECT_EQ(analysis0.position.y, 3.0);
    EXPECT_EQ(analysis0.position.z, 4.0);
    EXPECT_EQ(analysis0.direction.x, 5.0);
    EXPECT_EQ(analysis0.direction.y, 6.0);
    EXPECT_EQ(analysis0.direction.z, 7.0);
    EXPECT_EQ(analysis0.objectClass, "a");
    EXPECT_EQ(analysis0.objectConfidence, 0.75);
    EXPECT_EQ(analysis0.objectClassProbability, 0.5);
    EXPECT_EQ(analysis0.boundingBox.center.x, 8.0);
    EXPECT_EQ(analysis0.boundingBox.center.y, 9.0);
    EXPECT_EQ(analysis0.boundingBox.width, 10.0);
    EXPECT_EQ(analysis0.boundingBox.height, 11.0);
    EXPECT_EQ(analysis0.personPoseImage, nullopt);
    EXPECT_EQ(analysis0.personPose, nullopt);
    EXPECT_EQ(analysis0.personPoseConfidence, nullopt);
    EXPECT_EQ(analysis0.faceDescriptor, nullopt);
    EXPECT_EQ(analysis0.faceAlignmentKeypointCount, nullopt);
    EXPECT_EQ(analysis0.faceSharpnessScore, nullopt);

    VideoAnalysis analysis1{
        Timestamp(10),
        Position{20.0, 30.0, 40.0},
        Direction{50.0, 60.0, 70.0},
        "b",
        0.5f,
        0.25f,
        BoundingBox{{80.0, 90.0}, 100.0, 110.0},
        {ImagePosition{120.0, 130.0}},
        {Position{140.0, 150.0, 160.0}},
        {0.5f}};
    EXPECT_EQ(analysis1.timestamp.unixEpochMs, 10);
    EXPECT_EQ(analysis1.position.x, 20.0);
    EXPECT_EQ(analysis1.position.y, 30.0);
    EXPECT_EQ(analysis1.position.z, 40.0);
    EXPECT_EQ(analysis1.direction.x, 50.0);
    EXPECT_EQ(analysis1.direction.y, 60.0);
    EXPECT_EQ(analysis1.direction.z, 70.0);
    EXPECT_EQ(analysis1.objectClass, "b");
    EXPECT_EQ(analysis1.objectConfidence, 0.5f);
    EXPECT_EQ(analysis1.objectClassProbability, 0.25f);
    EXPECT_EQ(analysis1.boundingBox.center.x, 80.0);
    EXPECT_EQ(analysis1.boundingBox.center.y, 90.0);
    EXPECT_EQ(analysis1.boundingBox.width, 100.0);
    EXPECT_EQ(analysis1.boundingBox.height, 110.0);
    EXPECT_EQ(analysis1.personPoseImage, vector<ImagePosition>({ImagePosition{120.0, 130.0}}));
    EXPECT_EQ(analysis1.personPose, vector<Position>({Position{140.0, 150.0, 160.0}}));
    EXPECT_EQ(analysis1.personPoseConfidence, vector<float>({0.5}));
    EXPECT_EQ(analysis1.faceDescriptor, nullopt);
    EXPECT_EQ(analysis1.faceAlignmentKeypointCount, nullopt);
    EXPECT_EQ(analysis1.faceSharpnessScore, nullopt);

    VideoAnalysis analysis2{
        Timestamp(100),
        Position{200.0, 300.0, 400.0},
        Direction{500.0, 600.0, 700.0},
        "c",
        0.25f,
        0.125f,
        BoundingBox{{800.0, 900.0}, 1000.0, 1100.0},
        {ImagePosition{1200.0, 1300.0}},
        {Position{1400.0, 1500.0, 1600.0}},
        {0.5f},
        {200.f},
        5,
        0.625f};
    EXPECT_EQ(analysis2.timestamp.unixEpochMs, 100);
    EXPECT_EQ(analysis2.position.x, 200.0);
    EXPECT_EQ(analysis2.position.y, 300.0);
    EXPECT_EQ(analysis2.position.z, 400.0);
    EXPECT_EQ(analysis2.direction.x, 500.0);
    EXPECT_EQ(analysis2.direction.y, 600.0);
    EXPECT_EQ(analysis2.direction.z, 700.0);
    EXPECT_EQ(analysis2.objectClass, "c");
    EXPECT_EQ(analysis2.objectConfidence, 0.25);
    EXPECT_EQ(analysis2.objectClassProbability, 0.125);
    EXPECT_EQ(analysis2.boundingBox.center.x, 800.0);
    EXPECT_EQ(analysis2.boundingBox.center.y, 900.0);
    EXPECT_EQ(analysis2.boundingBox.width, 1000.0);
    EXPECT_EQ(analysis2.boundingBox.height, 1100.0);
    EXPECT_EQ(analysis2.personPoseImage, vector<ImagePosition>({ImagePosition{1200.0, 1300.0}}));
    EXPECT_EQ(analysis2.personPose, vector<Position>({Position{1400.0, 1500.0, 1600.0}}));
    EXPECT_EQ(analysis2.personPoseConfidence, vector<float>({0.5f}));
    EXPECT_EQ(analysis2.faceDescriptor, vector<float>({200.f}));
    EXPECT_EQ(analysis2.faceAlignmentKeypointCount, 5);
    EXPECT_EQ(analysis2.faceSharpnessScore, 0.625);
}
