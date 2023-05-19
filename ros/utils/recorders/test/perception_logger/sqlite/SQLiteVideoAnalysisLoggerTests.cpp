#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>

#include "SQLitePerceptionLoggerTests.h"

#include <gtest/gtest.h>

#include "comparisons.h"

using namespace std;

void readVideoAnalysis(
    SQLite::Database& database,
    int64_t id,
    string& objectClass,
    BoundingBox& boundingBox,
    vector<ImagePosition>& personPoseImage,
    vector<Position>& personPose,
    vector<float>& personPoseConfidence,
    vector<float>& faceDescriptor)
{
    SQLite::Statement query(
        database,
        "SELECT object_class, bounding_box_centre_x, bounding_box_centre_y, bounding_box_width, bounding_box_height, "
        "person_pose_image, person_pose, person_pose_confidence, face_descriptor FROM video_analysis WHERE "
        "perception_id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        objectClass = "";
        boundingBox = {};
        personPoseImage.clear();
        personPose.clear();
        personPoseConfidence.clear();
        faceDescriptor.clear();
        return;
    }

    objectClass = query.getColumn(0).getString();
    boundingBox.center.x = query.getColumn(1).getDouble();
    boundingBox.center.y = query.getColumn(2).getDouble();
    boundingBox.width = query.getColumn(3).getDouble();
    boundingBox.height = query.getColumn(4).getDouble();
    columnToVector(query.getColumn(5), personPoseImage);
    columnToVector(query.getColumn(6), personPose);
    columnToVector(query.getColumn(7), personPoseConfidence);
    columnToVector(query.getColumn(8), faceDescriptor);
}

TEST(SQLiteVideoAnalysisLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteVideoAnalysisLogger logger(database);

    int64_t id0 = logger.log(
        VideoAnalysis(Timestamp(101), Position{1, 2, 3}, Direction{4, 5, 6}, "banana", BoundingBox{{7, 8}, 9, 10}));
    int64_t id1 = logger.log(VideoAnalysis(
        Timestamp(102),
        Position{11, 12, 13},
        Direction{14, 15, 16},
        "person",
        BoundingBox{{17, 18}, 19, 20},
        {{21, 22}},
        {{23, 24, 25}},
        {0.5}));
    int64_t id2 = logger.log(VideoAnalysis(
        Timestamp(103),
        Position{26, 27, 28},
        Direction{29, 30, 31},
        "person",
        BoundingBox{{32, 33}, 34, 35},
        {{36, 37}},
        {{38, 39, 40}},
        {0.75},
        {41, 42}));

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101), Position{1, 2, 3}, Direction{4, 5, 6}));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102), Position{11, 12, 13}, Direction{14, 15, 16}));
    EXPECT_TRUE(perceptionExists(database, id2, Timestamp(103), Position{26, 27, 28}, Direction{29, 30, 31}));

    string objectClass;
    BoundingBox boundingBox;
    vector<ImagePosition> personPoseImage;
    vector<Position> personPose;
    vector<float> personPoseConfidence;
    vector<float> faceDescriptor;

    readVideoAnalysis(
        database,
        id0,
        objectClass,
        boundingBox,
        personPoseImage,
        personPose,
        personPoseConfidence,
        faceDescriptor);
    EXPECT_EQ(objectClass, "banana");
    EXPECT_EQ(boundingBox, (BoundingBox{{7, 8}, 9, 10}));
    EXPECT_EQ(personPoseImage, vector<ImagePosition>({}));
    EXPECT_EQ(personPose, vector<Position>({}));
    EXPECT_EQ(personPoseConfidence, vector<float>({}));
    EXPECT_EQ(faceDescriptor, vector<float>({}));

    readVideoAnalysis(
        database,
        id1,
        objectClass,
        boundingBox,
        personPoseImage,
        personPose,
        personPoseConfidence,
        faceDescriptor);
    EXPECT_EQ(objectClass, "person");
    EXPECT_EQ(boundingBox, (BoundingBox{{17, 18}, 19, 20}));
    ASSERT_EQ(personPoseImage, vector<ImagePosition>({ImagePosition{21, 22}}));
    ASSERT_EQ(personPose, vector<Position>({Position{23, 24, 25}}));
    EXPECT_EQ(personPoseConfidence, vector<float>({0.5f}));
    EXPECT_EQ(faceDescriptor, vector<float>({}));

    readVideoAnalysis(
        database,
        id2,
        objectClass,
        boundingBox,
        personPoseImage,
        personPose,
        personPoseConfidence,
        faceDescriptor);
    EXPECT_EQ(objectClass, "person");
    EXPECT_EQ(boundingBox, (BoundingBox{{32, 33}, 34, 35}));
    ASSERT_EQ(personPoseImage, vector<ImagePosition>({ImagePosition{36, 37}}));
    ASSERT_EQ(personPose, vector<Position>({Position{38, 39, 40}}));
    EXPECT_EQ(personPoseConfidence, vector<float>({0.75f}));
    EXPECT_EQ(faceDescriptor, vector<float>({41.f, 42.f}));
}
