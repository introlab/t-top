#include <home_logger_common/commands/CommandParser.h>

#include <gtest/gtest.h>

using namespace std;

TEST(CommandTests, weatherCommand_constructor_shouldSetAttributes)
{
    WeatherCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<WeatherCommand>());
    EXPECT_EQ(testee0.time(), tl::nullopt);

    WeatherCommand testee1("t1", WeatherTime::WEEK);
    EXPECT_EQ(testee1.transcript(), "t1");
    EXPECT_EQ(testee0.type(), CommandType::get<WeatherCommand>());
    EXPECT_EQ(testee1.time(), WeatherTime::WEEK);
}

TEST(CommandTests, increaseVolumeCommand_constructor_shouldSetAttributes)
{
    IncreaseVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<IncreaseVolumeCommand>());
}

TEST(CommandTests, decreaseVolumeCommand_constructor_shouldSetAttributes)
{
    DecreaseVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<DecreaseVolumeCommand>());
}

TEST(CommandTests, muteCommand_constructor_shouldSetAttributes)
{
    MuteCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<MuteCommand>());
}

TEST(CommandTests, unmuteCommand_constructor_shouldSetAttributes)
{
    UnmuteCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<UnmuteCommand>());
}

TEST(CommandTests, setVolumeCommand_constructor_shouldSetAttributes)
{
    SetVolumeCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<SetVolumeCommand>());
    EXPECT_EQ(testee0.volumePercent(), tl::nullopt);

    SetVolumeCommand testee1("t1", 2.5f);
    EXPECT_EQ(testee1.transcript(), "t1");
    EXPECT_EQ(testee1.type(), CommandType::get<SetVolumeCommand>());
    EXPECT_EQ(testee1.volumePercent(), 2.5f);
}

TEST(CommandTests, getVolumeCommand_constructor_shouldSetAttributes)
{
    GetVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<GetVolumeCommand>());
}

TEST(CommandTests, sleepCommand_constructor_shouldSetAttributes)
{
    SleepCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<SleepCommand>());
}

TEST(CommandTests, currentDateCommand_constructor_shouldSetAttributes)
{
    CurrentDateCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentDateCommand>());
}

TEST(CommandTests, currentTimeCommand_constructor_shouldSetAttributes)
{
    CurrentTimeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentTimeCommand>());
}

TEST(CommandTests, currentDateTimeCommand_constructor_shouldSetAttributes)
{
    CurrentDateTimeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentDateTimeCommand>());
}

TEST(CommandTests, addAlarmCommand_constructor_shouldSetAttributes)
{
    AddAlarmCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_EQ(testee0.alarmType(), tl::nullopt);
    EXPECT_EQ(testee0.frequency(), tl::nullopt);
    EXPECT_EQ(testee0.weekDay(), tl::nullopt);
    EXPECT_EQ(testee0.date(), tl::nullopt);
    EXPECT_EQ(testee0.time(), tl::nullopt);

    AddAlarmCommand testee1("t0", AlarmType::PUNCTUAL, AlarmFrequency::WEEKLY, 1, Date(2022, 7, 3), Time(10, 40));
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_EQ(testee1.alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(testee1.frequency(), AlarmFrequency::WEEKLY);
    EXPECT_EQ(testee1.weekDay(), 1);
    EXPECT_EQ(testee1.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee1.time(), Time(10, 40));
}

TEST(CommandTests, listAlarmsCommand_constructor_shouldSetAttributes)
{
    ListAlarmsCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListAlarmsCommand>());
}

TEST(CommandTests, removeAlarmCommand_constructor_shouldSetAttributes)
{
    RemoveAlarmCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<RemoveAlarmCommand>());
    EXPECT_EQ(testee0.id(), tl::nullopt);

    RemoveAlarmCommand testee1("t0", 10);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<RemoveAlarmCommand>());
    EXPECT_EQ(testee1.id(), 10);
}

TEST(CommandTests, addReminderCommand_constructor_shouldSetAttributes)
{
    AddReminderCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<AddReminderCommand>());
    EXPECT_EQ(testee0.text(), tl::nullopt);
    EXPECT_EQ(testee0.datetime(), tl::nullopt);

    AddReminderCommand testee1("t0", "Hello", DateTime(2022, 7, 3, 10, 40));
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<AddReminderCommand>());
    EXPECT_EQ(testee1.text(), "Hello");
    EXPECT_EQ(testee1.datetime(), DateTime(2022, 7, 3, 10, 40));
}

TEST(CommandTests, listRemindersCommand_constructor_shouldSetAttributes)
{
    ListRemindersCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListRemindersCommand>());
}

TEST(CommandTests, removeReminderCommand_constructor_shouldSetAttributes)
{
    RemoveReminderCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<RemoveReminderCommand>());
    EXPECT_EQ(testee0.id(), tl::nullopt);

    RemoveReminderCommand testee1("t0", 10);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<RemoveReminderCommand>());
    EXPECT_EQ(testee1.id(), 10);
}

TEST(CommandTests, listCommandsCommand_constructor_shouldSetAttributes)
{
    ListCommandsCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListCommandsCommand>());
}

TEST(CommandTests, nothingCommand_constructor_shouldSetAttributes)
{
    NothingCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<NothingCommand>());
}
