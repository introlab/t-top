#include <home_logger_common/commands/CommandParsers.h>

#include <gtest/gtest.h>

using namespace std;

TEST(CommandParserTests, keywordParse_emptyString_shouldReturnNull)
{
    KeywordCommandParser<SleepCommand> testee({{"weather"}, {"actual", "current", "present"}}, {"volume", "sleep"});
    EXPECT_EQ(testee.parse(""), nullptr);
}

TEST(CommandParserTests, keywordParse_oneKeyword_shouldReturnNull)
{
    KeywordCommandParser<SleepCommand> testee({{"weatHer"}, {"actual", "current", "present"}}, {"voLume", "slEep"});
    EXPECT_EQ(testee.parse("tell me the weaTher"), nullptr);
}

TEST(CommandParserTests, keywordParse_twoKeywords_shouldNotReturnNull)
{
    KeywordCommandParser<SleepCommand> testee({{"WEAther"}, {"actuaL", "curRent", "preSent"}}, {"volume", "sleep"});
    EXPECT_NE(testee.parse("tell me the actuaL weaTher"), nullptr);
    EXPECT_NE(testee.parse("tell me the currenT weatHer"), nullptr);
    EXPECT_NE(testee.parse("tell me the presenT weathEr"), nullptr);
}

TEST(CommandParserTests, keywordParse_twoKeywordsAndOneNonKeyword_shouldNotReturnNull)
{
    KeywordCommandParser<SleepCommand> testee({{"weather"}, {"actual", "current", "present"}}, {"volumE", "sleeP"});
    EXPECT_EQ(testee.parse("tell me the actual weather, vOlume"), nullptr);
    EXPECT_EQ(testee.parse("tell me the current weather, sleEp"), nullptr);
    EXPECT_EQ(testee.parse("tell me the present weather, voLume sLeep"), nullptr);
}

TEST(CommandParserTests, keywordParse_twoKeywords_shoulSetTheTranscript)
{
    KeywordCommandParser<SleepCommand> testee({{"weather"}, {"actual", "current", "present"}}, {"volume", "sleep"});
    string transcript = "tell me the actual weather";
    unique_ptr<Command> command = testee.parse(transcript);

    EXPECT_NE(command, nullptr);
    EXPECT_EQ(command->transcript(), transcript);
    EXPECT_EQ(command->type(), CommandType::get<SleepCommand>());
}

TEST(CommandParserTests, keywordParse_empty_shoulShouldIgnore)
{
    KeywordCommandParser<SleepCommand> testee({{}, {"actual", "current", "present"}}, {"volume", "sleep"});
    EXPECT_NE(testee.parse("tell me the actual"), nullptr);
}
