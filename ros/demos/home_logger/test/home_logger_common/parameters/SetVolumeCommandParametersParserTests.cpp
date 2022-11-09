#include <home_logger_common/parameters/SetVolumeCommandParametersParser.h>

#include <gtest/gtest.h>

using namespace std;

TEST(SetVolumeCommandParametersParserTests, parse_nonNullFaceDescriptor_shouldThrowRuntimeError)
{
    SetVolumeCommandParametersParser testee;
    auto command = make_shared<SetVolumeCommand>("");
    EXPECT_THROW(testee.parse(command, nullopt, nullopt, FaceDescriptor({})), runtime_error);
}

TEST(SetVolumeCommandParametersParserTests, parse_invalidParameterName_shouldThrowRuntimeError)
{
    SetVolumeCommandParametersParser testee;
    auto command = make_shared<SetVolumeCommand>("");
    EXPECT_THROW(testee.parse(command, "a", "b", nullopt), runtime_error);
}

TEST(SetVolumeCommandParametersParserTests, parse_transcriptNothing_shouldReturnACopy)
{
    SetVolumeCommandParametersParser testee;
    auto command = testee.parse(make_shared<SetVolumeCommand>("a"), nullopt, nullopt, nullopt);
    auto setVolumeCommand = dynamic_pointer_cast<SetVolumeCommand>(command);
    EXPECT_EQ(setVolumeCommand->transcript(), "a");
    EXPECT_EQ(setVolumeCommand->volumePercent(), nullopt);
}

TEST(SetVolumeCommandParametersParserTests, parse_parameterNothing_shouldReturnACopy)
{
    SetVolumeCommandParametersParser testee;
    auto command = testee.parse(make_shared<SetVolumeCommand>("a"), "volume", "", nullopt);
    auto setVolumeCommand = dynamic_pointer_cast<SetVolumeCommand>(command);
    EXPECT_EQ(setVolumeCommand->transcript(), "a");
    EXPECT_EQ(setVolumeCommand->volumePercent(), nullopt);
}

TEST(SetVolumeCommandParametersParserTests, parse_transcriptCurrent_shouldReturnACopyWithVolume)
{
    SetVolumeCommandParametersParser testee;
    auto command = testee.parse(make_shared<SetVolumeCommand>("15"), nullopt, nullopt, nullopt);
    auto setVolumeCommand = dynamic_pointer_cast<SetVolumeCommand>(command);
    EXPECT_EQ(setVolumeCommand->transcript(), "15");
    EXPECT_EQ(setVolumeCommand->volumePercent(), 15);
}

TEST(SetVolumeCommandParametersParserTests, parse_parameterCurrent_shouldReturnACopyWithVolume)
{
    SetVolumeCommandParametersParser testee;
    auto command = testee.parse(make_shared<SetVolumeCommand>("a"), "volume", "26", nullopt);
    auto setVolumeCommand = dynamic_pointer_cast<SetVolumeCommand>(command);
    EXPECT_EQ(setVolumeCommand->transcript(), "a");
    EXPECT_EQ(setVolumeCommand->volumePercent(), 26);
}
