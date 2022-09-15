#include <home_logger_common/parameters/WeatherCommandParametersParser.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

class WeatherCommandParametersParserFrenchTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadFrenchStringResources(); }
};

class WeatherCommandParametersParserEnglishTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadEnglishStringResources(); }
};

TEST_F(WeatherCommandParametersParserFrenchTests, parse_nullParameterNameOrParameterResponse_shouldThrowRuntimeError)
{
    WeatherCommandParametersParser testee;
    auto command = make_shared<WeatherCommand>("");
    EXPECT_THROW(testee.parse(command, "a", tl::nullopt, tl::nullopt), runtime_error);
    EXPECT_THROW(testee.parse(command, tl::nullopt, "a", tl::nullopt), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    WeatherCommandParametersParser testee;
    auto command = make_shared<WeatherCommand>("");
    EXPECT_THROW(testee.parse(command, tl::nullopt, tl::nullopt, FaceDescriptor({})), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    WeatherCommandParametersParser testee;
    auto command = make_shared<WeatherCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", tl::nullopt), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptNothing_shouldReturnACopy)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), tl::nullopt);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterNothing_shouldReturnACopy)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), tl::nullopt);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("Actuelle"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "Actuelle");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "Présente", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("aujourd'hUi"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "aujourd'hUi");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("b"), "time", "aujourD'hui", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "b");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("deMain"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "deMain");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "DEMAIN", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("semaine"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "semaine");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "semAine", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}


TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("actual"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "actual");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "Current", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("today"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "today");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "today", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("tomorroW"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "tomorroW");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "tomorrow", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("week"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "week");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "week's weather forecast", tl::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}
