#include <home_logger_common/parameters/AlarmCommandParametersParsers.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

class AddAlarmCommandParametersParserFrenchTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadFrenchStringResources(); }
};

class AddAlarmCommandParametersParserEnglishTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadEnglishStringResources(); }
};

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    AddAlarmCommandParametersParser testee;
    auto command = make_shared<AddAlarmCommand>("");
    EXPECT_THROW(testee.parse(command, tl::nullopt, tl::nullopt, FaceDescriptor({})), runtime_error);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    AddAlarmCommandParametersParser testee;
    auto command = make_shared<AddAlarmCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", tl::nullopt), runtime_error);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_transcriptNothing_shouldReturnACopy)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterNothing_shouldReturnACopy)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_transcriptPunctual_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("ponctuelle"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "ponctuelle");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterPunctual_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "ponctuElle", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_transcriptDaily_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("jour"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "jour");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::DAILY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterDaily_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "jOur", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::DAILY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_transcripWeekly_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("semAine"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "semAine");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterWeekly_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "semAine", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterWeekDay_shouldReturnACopyWithWeekDay)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "week_day", "mardi", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), 2);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterDate_shouldReturnACopyWithDate)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "date", "1 janvier 2022", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), Date(2022, 0, 1));
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserFrenchTests, parse_parameterTime_shouldReturnACopyWithTime)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "time", "17:30", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), Time(17, 30));
}


TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_transcriptPunctual_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("one-time"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "one-time");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterPunctual_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "punctUal", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_transcriptDaily_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("each day"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "each day");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::DAILY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterDaily_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "daIly", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::DAILY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_transcripWeekly_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("each week"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "each week");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterWeekly_shouldReturnACopyWithType)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "type", "Weekly", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterWeekDay_shouldReturnACopyWithWeekDay)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "week_day", "Sunday", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), 0);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterDate_shouldReturnACopyWithDate)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "date", "January 1st, 2022", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), Date(2022, 0, 1));
    EXPECT_EQ(addAlarmCommand->time(), tl::nullopt);
}

TEST_F(AddAlarmCommandParametersParserEnglishTests, parse_parameterTime_shouldReturnACopyWithTime)
{
    AddAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddAlarmCommand>("a"), "time", "5:30 PM", tl::nullopt);
    auto addAlarmCommand = dynamic_pointer_cast<AddAlarmCommand>(command);
    EXPECT_EQ(addAlarmCommand->transcript(), "a");
    EXPECT_EQ(addAlarmCommand->alarmType(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->weekDay(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->date(), tl::nullopt);
    EXPECT_EQ(addAlarmCommand->time(), Time(17, 30));
}


TEST(RemoveAlarmCommandParametersParserTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = make_shared<RemoveAlarmCommand>("");
    EXPECT_THROW(testee.parse(command, tl::nullopt, tl::nullopt, FaceDescriptor({})), runtime_error);
}

TEST(RemoveAlarmCommandParametersParserTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = make_shared<RemoveAlarmCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", tl::nullopt), runtime_error);
}

TEST(RemoveAlarmCommandParametersParserTests, parse_transcriptNothing_shouldReturnACopy)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveAlarmCommand>("a"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto removeAlarmCommand = dynamic_pointer_cast<RemoveAlarmCommand>(command);
    EXPECT_EQ(removeAlarmCommand->transcript(), "a");
    EXPECT_EQ(removeAlarmCommand->id(), tl::nullopt);
}

TEST(RemoveAlarmCommandParametersParserTests, parse_parameterNothing_shouldReturnACopy)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveAlarmCommand>("a"), "id", "", tl::nullopt);
    auto removeAlarmCommand = dynamic_pointer_cast<RemoveAlarmCommand>(command);
    EXPECT_EQ(removeAlarmCommand->transcript(), "a");
    EXPECT_EQ(removeAlarmCommand->id(), tl::nullopt);
}

TEST(RemoveAlarmCommandParametersParserTests, parse_transcriptCurrent_shouldReturnACopyWithVolume)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveAlarmCommand>("Alarme 15"), tl::nullopt, tl::nullopt, tl::nullopt);
    auto removeAlarmCommand = dynamic_pointer_cast<RemoveAlarmCommand>(command);
    EXPECT_EQ(removeAlarmCommand->transcript(), "Alarme 15");
    EXPECT_EQ(removeAlarmCommand->id(), 15);
}

TEST(RemoveAlarmCommandParametersParserTests, parse_parameterCurrent_shouldReturnACopyWithVolume)
{
    RemoveAlarmCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveAlarmCommand>("a"), "id", "Alarme 26", tl::nullopt);
    auto removeAlarmCommand = dynamic_pointer_cast<RemoveAlarmCommand>(command);
    EXPECT_EQ(removeAlarmCommand->transcript(), "a");
    EXPECT_EQ(removeAlarmCommand->id(), 26);
}
