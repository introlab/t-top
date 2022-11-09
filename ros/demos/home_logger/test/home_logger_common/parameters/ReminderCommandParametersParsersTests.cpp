#include <home_logger_common/parameters/ReminderCommandParametersParsers.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

TEST(AddReminderCommandParametersParserTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    AddReminderCommandParametersParser testee;
    auto command = make_shared<AddReminderCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", nullopt), runtime_error);
}

TEST(AddReminderCommandParametersParserTests, parse_transcriptNothing_shouldReturnACopy)
{
    AddReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddReminderCommand>("a"), nullopt, nullopt, nullopt);
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), nullopt);
    EXPECT_EQ(addReminderCommand->date(), nullopt);
    EXPECT_EQ(addReminderCommand->time(), nullopt);
    EXPECT_EQ(addReminderCommand->faceDescriptor(), nullopt);
}

TEST(AddReminderCommandParametersParserTests, parse_parameterNothing_shouldReturnACopy)
{
    AddReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddReminderCommand>("a"), "date", "", nullopt);
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), nullopt);
    EXPECT_EQ(addReminderCommand->date(), nullopt);
    EXPECT_EQ(addReminderCommand->time(), nullopt);
    EXPECT_EQ(addReminderCommand->faceDescriptor(), nullopt);
}

TEST(AddReminderCommandParametersParserTests, parse_parameterText_shouldReturnACopyWithText)
{
    AddReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddReminderCommand>("a"), "text", "Faire la tondeuse", nullopt);
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), "Faire la tondeuse");
    EXPECT_EQ(addReminderCommand->date(), nullopt);
    EXPECT_EQ(addReminderCommand->time(), nullopt);
    EXPECT_EQ(addReminderCommand->faceDescriptor(), nullopt);
}

TEST(AddReminderCommandParametersParserTests, parse_parameterDate_shouldReturnACopyWithDate)
{
    loadFrenchStringResources();

    AddReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddReminderCommand>("a"), "date", "1 mars 2023", nullopt);
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), nullopt);
    EXPECT_EQ(addReminderCommand->date(), Date(2023, 2, 1));
    EXPECT_EQ(addReminderCommand->time(), nullopt);
    EXPECT_EQ(addReminderCommand->faceDescriptor(), nullopt);
}

TEST(AddReminderCommandParametersParserTests, parse_parameterText_shouldReturnACopyWithTime)
{
    loadFrenchStringResources();

    AddReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<AddReminderCommand>("a"), "time", "19h56", nullopt);
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), nullopt);
    EXPECT_EQ(addReminderCommand->date(), nullopt);
    EXPECT_EQ(addReminderCommand->time(), Time(19, 56));
    EXPECT_EQ(addReminderCommand->faceDescriptor(), nullopt);
}

TEST(AddReminderCommandParametersParserTests, parse_faceDescriptor_shouldReturnACopyWithFaceDescriptor)
{
    loadFrenchStringResources();

    AddReminderCommandParametersParser testee;
    auto command =
        testee.parse(make_shared<AddReminderCommand>("a"), nullopt, nullopt, FaceDescriptor({11.f}));
    auto addReminderCommand = dynamic_pointer_cast<AddReminderCommand>(command);
    EXPECT_EQ(addReminderCommand->transcript(), "a");
    EXPECT_EQ(addReminderCommand->text(), nullopt);
    EXPECT_EQ(addReminderCommand->date(), nullopt);
    EXPECT_EQ(addReminderCommand->time(), nullopt);
    EXPECT_TRUE(addReminderCommand->faceDescriptor().has_value());
    EXPECT_EQ(addReminderCommand->faceDescriptor().value().data(), vector<float>({11.f}));
}


TEST(RemoveReminderCommandParametersParserTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    RemoveReminderCommandParametersParser testee;
    auto command = make_shared<RemoveReminderCommand>("");
    EXPECT_THROW(testee.parse(command, nullopt, nullopt, FaceDescriptor({})), runtime_error);
}

TEST(RemoveReminderCommandParametersParserTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    RemoveReminderCommandParametersParser testee;
    auto command = make_shared<RemoveReminderCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", nullopt), runtime_error);
}

TEST(RemoveReminderCommandParametersParserTests, parse_transcriptNothing_shouldReturnACopy)
{
    RemoveReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveReminderCommand>("a"), nullopt, nullopt, nullopt);
    auto removeReminderCommand = dynamic_pointer_cast<RemoveReminderCommand>(command);
    EXPECT_EQ(removeReminderCommand->transcript(), "a");
    EXPECT_EQ(removeReminderCommand->id(), nullopt);
}

TEST(RemoveReminderCommandParametersParserTests, parse_parameterNothing_shouldReturnACopy)
{
    RemoveReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveReminderCommand>("a"), "id", "", nullopt);
    auto removeReminderCommand = dynamic_pointer_cast<RemoveReminderCommand>(command);
    EXPECT_EQ(removeReminderCommand->transcript(), "a");
    EXPECT_EQ(removeReminderCommand->id(), nullopt);
}

TEST(RemoveReminderCommandParametersParserTests, parse_transcriptCurrent_shouldReturnACopyWithVolume)
{
    RemoveReminderCommandParametersParser testee;
    auto command =
        testee.parse(make_shared<RemoveReminderCommand>("Alarme 15"), nullopt, nullopt, nullopt);
    auto removeReminderCommand = dynamic_pointer_cast<RemoveReminderCommand>(command);
    EXPECT_EQ(removeReminderCommand->transcript(), "Alarme 15");
    EXPECT_EQ(removeReminderCommand->id(), 15);
}

TEST(RemoveReminderCommandParametersParserTests, parse_parameterCurrent_shouldReturnACopyWithVolume)
{
    RemoveReminderCommandParametersParser testee;
    auto command = testee.parse(make_shared<RemoveReminderCommand>("a"), "id", "Alarme 26", nullopt);
    auto removeReminderCommand = dynamic_pointer_cast<RemoveReminderCommand>(command);
    EXPECT_EQ(removeReminderCommand->transcript(), "a");
    EXPECT_EQ(removeReminderCommand->id(), 26);
}
