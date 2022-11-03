#include <perception_logger/SpeechLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(SpeechLoggerTests, speechSourceToString_shouldConvertTheSpeechSource)
{
    EXPECT_EQ(speechSourceToString(SpeechSource::ROBOT), "robot");
    EXPECT_EQ(speechSourceToString(SpeechSource::HUMAN), "human");
}

TEST(SpeechLoggerTests, speech_constructor_shouldSetAttributes)
{
    Speech speech(Timestamp(1), SpeechSource::ROBOT, "ab");
    EXPECT_EQ(speech.timestamp.unixEpochMs, 1);
    EXPECT_EQ(speech.source, SpeechSource::ROBOT);
    EXPECT_EQ(speech.text, "ab");
}
