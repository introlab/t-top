#include <perception_logger/sqlite/SQLiteSpeechLogger.h>

#include "SQLitePerceptionLoggerTests.h"

#include <gtest/gtest.h>

using namespace std;

void readSpeech(SQLite::Database& database, int64_t id, string& source, string& text)
{
    SQLite::Statement query(database, "SELECT source, text FROM speech WHERE perception_id=?");

    query.bind(1, id);
    if (!query.executeStep())
    {
        source = "";
        text = "";
        return;
    }

    source = query.getColumn(0).getString();
    text = query.getColumn(1).getString();
}

TEST(SQLiteSpeechLoggerTests, log_shouldInsertAndReturnId)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    SQLiteSpeechLogger logger(database);

    int64_t id0 = logger.log(Speech(Timestamp(101), SpeechSource::ROBOT, "bob"));
    int64_t id1 = logger.log(Speech(Timestamp(102), SpeechSource::HUMAN, "marc"));

    EXPECT_TRUE(perceptionExists(database, id0, Timestamp(101)));
    EXPECT_TRUE(perceptionExists(database, id1, Timestamp(102)));

    string source;
    string text;

    readSpeech(database, id0, source, text);
    EXPECT_EQ(source, "robot");
    EXPECT_EQ(text, "bob");

    readSpeech(database, id1, source, text);
    EXPECT_EQ(source, "human");
    EXPECT_EQ(text, "marc");
}
