#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>

#include "SQLitePerceptionLoggerTests.h"

#include <gtest/gtest.h>

using namespace std;

void readAudioAnalysis(
    SQLite::Database& database,
    int64_t id,
    int64_t& trackingId,
    string& classes,
    vector<float>& voiceDescriptor)
{
    SQLite::Statement query(
        database,
        "SELECT tracking_id, classes, voice_descriptor FROM audio_analysis WHERE perception_id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        classes = "";
        voiceDescriptor.clear();
        return;
    }

    trackingId = query.getColumn(0).getInt64();
    classes = query.getColumn(1).getString();
    columnToVector(query.getColumn(2), voiceDescriptor);
}

TEST(SQLiteAudioAnalysisLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteAudioAnalysisLogger logger(database);

    int64_t id0 = logger.log(AudioAnalysis(Timestamp(101), Direction{1, 2, 3}, 4, "music,water"));
    int64_t id1 = logger.log(AudioAnalysis(Timestamp(102), Direction{4, 5, 6}, 7, "voice", {8.f, 9.f}));

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101), Direction{1, 2, 3}));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102), Direction{4, 5, 6}));

    int64_t trackingId;
    string classes;
    vector<float> voiceDescriptor;

    readAudioAnalysis(database, id0, trackingId, classes, voiceDescriptor);
    EXPECT_EQ(trackingId, 4);
    EXPECT_EQ(classes, "music,water");
    EXPECT_EQ(voiceDescriptor, vector<float>({}));

    readAudioAnalysis(database, id1, trackingId, classes, voiceDescriptor);
    EXPECT_EQ(trackingId, 7);
    EXPECT_EQ(classes, "voice");
    EXPECT_EQ(voiceDescriptor, vector<float>({8.f, 9.f}));
}
