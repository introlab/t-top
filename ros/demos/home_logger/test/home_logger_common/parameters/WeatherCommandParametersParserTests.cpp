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
    EXPECT_THROW(testee.parse(command, "a", std::nullopt, std::nullopt), runtime_error);
    EXPECT_THROW(testee.parse(command, std::nullopt, "a", std::nullopt), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    WeatherCommandParametersParser testee;
    auto command = make_shared<WeatherCommand>("");
    EXPECT_THROW(testee.parse(command, std::nullopt, std::nullopt, FaceDescriptor({})), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    WeatherCommandParametersParser testee;
    auto command = make_shared<WeatherCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", std::nullopt), runtime_error);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptNothing_shouldReturnACopy)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), std::nullopt);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterNothing_shouldReturnACopy)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), std::nullopt);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("Actuelle"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "Actuelle");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "Présente", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("aujourd'hUi"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "aujourd'hUi");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("b"), "time", "aujourD'hui", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "b");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("deMain"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "deMain");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "DEMAIN", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_transcriptWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("semaine"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "semaine");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}

TEST_F(WeatherCommandParametersParserFrenchTests, parse_parameterWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "semAine", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}


TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("actual"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "actual");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterCurrent_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "Current", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::CURRENT);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("today"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "today");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterToday_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "today", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TODAY);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("tomorroW"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "tomorroW");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterTomorrow_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "tomorrow", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::TOMORROW);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_transcriptWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("week"), std::nullopt, std::nullopt, std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "week");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}

TEST_F(WeatherCommandParametersParserEnglishTests, parse_parameterWeek_shouldReturnACopyWithTime)
{
    WeatherCommandParametersParser testee;
    auto command = testee.parse(make_shared<WeatherCommand>("a"), "time", "week's weather forecast", std::nullopt);
    auto weatherCommand = dynamic_pointer_cast<WeatherCommand>(command);
    EXPECT_EQ(weatherCommand->transcript(), "a");
    EXPECT_EQ(weatherCommand->time(), WeatherTime::WEEK);
}
