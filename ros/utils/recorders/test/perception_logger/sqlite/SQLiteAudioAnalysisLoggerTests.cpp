#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>

#include "SQLitePerceptionLoggerTests.h"

#include <gtest/gtest.h>

using namespace std;

void readAudioAnalysis(SQLite::Database& database, int64_t id, string& classes, vector<float>& voiceDescriptor)
{
    SQLite::Statement query(database, "SELECT classes, voice_descriptor FROM audio_analysis WHERE perception_id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        classes = "";
        voiceDescriptor.clear();
        return;
    }

    classes = query.getColumn(0).getString();
    columnToVector(query.getColumn(1), voiceDescriptor);
}

TEST(SQLiteAudioAnalysisLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteAudioAnalysisLogger logger(database);

    int64_t id0 = logger.log(AudioAnalysis(Timestamp(101), Direction{1, 2, 3}, "music,water"));
    int64_t id1 = logger.log(AudioAnalysis(Timestamp(102), Direction{4, 5, 6}, "voice", {7.f, 8.f}));

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101), Direction{1, 2, 3}));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102), Direction{4, 5, 6}));

    string classes;
    vector<float> voiceDescriptor;

    readAudioAnalysis(database, id0, classes, voiceDescriptor);
    EXPECT_EQ(classes, "music,water");
    EXPECT_EQ(voiceDescriptor, vector<float>({}));

    readAudioAnalysis(database, id1, classes, voiceDescriptor);
    EXPECT_EQ(classes, "voice");
    EXPECT_EQ(voiceDescriptor, vector<float>({7.f, 8.f}));
}
