#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>

#include "SQLitePerceptionLoggerTests.h"

#include <gtest/gtest.h>

using namespace std;

void readVideoAnalysis(
    SQLite::Database& database,
    int64_t id,
    string& objectClass,
    vector<Position>& personPose,
    vector<float>& personPoseConfidence,
    vector<float>& faceDescriptor)
{
    SQLite::Statement query(
        database,
        "SELECT object_class, person_pose, person_pose_confidence, face_descriptor FROM video_analysis WHERE "
        "perception_id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        objectClass = "";
        personPose.clear();
        personPoseConfidence.clear();
        faceDescriptor.clear();
        return;
    }

    objectClass = query.getColumn(0).getString();
    columnToVector(query.getColumn(1), personPose);
    columnToVector(query.getColumn(2), personPoseConfidence);
    columnToVector(query.getColumn(3), faceDescriptor);
}

TEST(SQLiteVideoAnalysisLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteVideoAnalysisLogger logger(database);

    int64_t id0 = logger.log(VideoAnalysis(Timestamp(101), Position(1, 2, 3), Direction(4, 5, 6), "banana"));
    int64_t id1 = logger.log(VideoAnalysis(
        Timestamp(102),
        Position(7, 8, 9),
        Direction(10, 11, 12),
        "person",
        {Position(13, 14, 15)},
        {0.5}));
    int64_t id2 = logger.log(VideoAnalysis(
        Timestamp(103),
        Position(16, 17, 18),
        Direction(19, 20, 21),
        "person",
        {Position(22, 23, 24)},
        {0.75},
        {7, 8}));

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101), Position(1, 2, 3), Direction(4, 5, 6)));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102), Position(7, 8, 9), Direction(10, 11, 12)));
    EXPECT_TRUE(perceptionExists(database, id2, Timestamp(103), Position(16, 17, 18), Direction(19, 20, 21)));

    string objectClass;
    vector<Position> personPose;
    vector<float> personPoseConfidence;
    vector<float> faceDescriptor;

    readVideoAnalysis(database, id0, objectClass, personPose, personPoseConfidence, faceDescriptor);
    EXPECT_EQ(objectClass, "banana");
    EXPECT_EQ(personPose, vector<Position>({}));
    EXPECT_EQ(personPoseConfidence, vector<float>({}));
    EXPECT_EQ(faceDescriptor, vector<float>({}));

    readVideoAnalysis(database, id1, objectClass, personPose, personPoseConfidence, faceDescriptor);
    EXPECT_EQ(objectClass, "person");
    EXPECT_EQ(personPose, vector<Position>({Position(13, 14, 15)}));
    EXPECT_EQ(personPoseConfidence, vector<float>({0.5}));
    EXPECT_EQ(faceDescriptor, vector<float>({}));

    readVideoAnalysis(database, id1, objectClass, personPose, personPoseConfidence, faceDescriptor);
    EXPECT_EQ(objectClass, "person");
    EXPECT_EQ(personPose, vector<Position>({Position(13, 14, 15)}));
    EXPECT_EQ(personPoseConfidence, vector<float>({0.5}));
    EXPECT_EQ(faceDescriptor, vector<float>({}));

    readVideoAnalysis(database, id2, objectClass, personPose, personPoseConfidence, faceDescriptor);
    EXPECT_EQ(objectClass, "person");
    EXPECT_EQ(personPose, vector<Position>({Position(22, 23, 24)}));
    EXPECT_EQ(personPoseConfidence, vector<float>({0.75}));
    EXPECT_EQ(faceDescriptor, vector<float>({7, 8}));
}
