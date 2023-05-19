#include <home_logger_common/commands/CommandParsers.h>

#include <gtest/gtest.h>

using namespace std;

TEST(CommandTests, weatherCommand_constructor_shouldSetAttributes)
{
    WeatherCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<WeatherCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.time(), nullopt);

    WeatherCommand testee1("t1", WeatherTime::WEEK);
    EXPECT_EQ(testee1.transcript(), "t1");
    EXPECT_EQ(testee0.type(), CommandType::get<WeatherCommand>());
    EXPECT_TRUE(testee1.isComplete());
    EXPECT_EQ(testee1.time(), WeatherTime::WEEK);
}

TEST(CommandTests, increaseVolumeCommand_constructor_shouldSetAttributes)
{
    IncreaseVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<IncreaseVolumeCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, decreaseVolumeCommand_constructor_shouldSetAttributes)
{
    DecreaseVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<DecreaseVolumeCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, setVolumeCommand_constructor_shouldSetAttributes)
{
    SetVolumeCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<SetVolumeCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.volumePercent(), nullopt);

    SetVolumeCommand testee1("t1", 2.5f);
    EXPECT_EQ(testee1.transcript(), "t1");
    EXPECT_EQ(testee1.type(), CommandType::get<SetVolumeCommand>());
    EXPECT_TRUE(testee1.isComplete());
    EXPECT_EQ(testee1.volumePercent(), 2.5f);
}

TEST(CommandTests, getVolumeCommand_constructor_shouldSetAttributes)
{
    GetVolumeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<GetVolumeCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, sleepCommand_constructor_shouldSetAttributes)
{
    SleepCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<SleepCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, currentDateCommand_constructor_shouldSetAttributes)
{
    CurrentDateCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentDateCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, currentTimeCommand_constructor_shouldSetAttributes)
{
    CurrentTimeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentTimeCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, currentDateTimeCommand_constructor_shouldSetAttributes)
{
    CurrentDateTimeCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<CurrentDateTimeCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, addAlarmCommand_constructor_shouldSetAttributes)
{
    AddAlarmCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.alarmType(), nullopt);
    EXPECT_EQ(testee0.weekDay(), nullopt);
    EXPECT_EQ(testee0.date(), nullopt);
    EXPECT_EQ(testee0.time(), nullopt);

    AddAlarmCommand testee1("t0", AlarmType::PUNCTUAL, nullopt, nullopt, nullopt);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee1.isComplete());
    EXPECT_EQ(testee1.alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(testee1.weekDay(), nullopt);
    EXPECT_EQ(testee1.date(), nullopt);
    EXPECT_EQ(testee1.time(), nullopt);

    AddAlarmCommand testee2("t0", AlarmType::PUNCTUAL, nullopt, Date(2022, 7, 3), nullopt);
    EXPECT_EQ(testee2.transcript(), "t0");
    EXPECT_EQ(testee2.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee2.isComplete());
    EXPECT_EQ(testee2.alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(testee2.weekDay(), nullopt);
    EXPECT_EQ(testee2.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee2.time(), nullopt);

    AddAlarmCommand testee3("t0", AlarmType::PUNCTUAL, nullopt, nullopt, Time(10, 40));
    EXPECT_EQ(testee3.transcript(), "t0");
    EXPECT_EQ(testee3.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee3.isComplete());
    EXPECT_EQ(testee3.alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(testee3.weekDay(), nullopt);
    EXPECT_EQ(testee3.date(), nullopt);
    EXPECT_EQ(testee3.time(), Time(10, 40));

    AddAlarmCommand testee4("t0", AlarmType::PUNCTUAL, nullopt, Date(2022, 7, 3), Time(10, 40));
    EXPECT_EQ(testee4.transcript(), "t0");
    EXPECT_EQ(testee4.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_TRUE(testee4.isComplete());
    EXPECT_EQ(testee4.alarmType(), AlarmType::PUNCTUAL);
    EXPECT_EQ(testee4.weekDay(), nullopt);
    EXPECT_EQ(testee4.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee4.time(), Time(10, 40));

    AddAlarmCommand testee5("t0", AlarmType::DAILY, nullopt, nullopt, nullopt);
    EXPECT_EQ(testee5.transcript(), "t0");
    EXPECT_EQ(testee5.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee5.isComplete());
    EXPECT_EQ(testee5.alarmType(), AlarmType::DAILY);
    EXPECT_EQ(testee5.weekDay(), nullopt);
    EXPECT_EQ(testee5.date(), nullopt);
    EXPECT_EQ(testee5.time(), nullopt);

    AddAlarmCommand testee6("t0", AlarmType::DAILY, nullopt, nullopt, Time(10, 40));
    EXPECT_EQ(testee6.transcript(), "t0");
    EXPECT_EQ(testee6.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_TRUE(testee6.isComplete());
    EXPECT_EQ(testee6.alarmType(), AlarmType::DAILY);
    EXPECT_EQ(testee6.weekDay(), nullopt);
    EXPECT_EQ(testee6.date(), nullopt);
    EXPECT_EQ(testee6.time(), Time(10, 40));

    AddAlarmCommand testee7("t0", AlarmType::WEEKLY, nullopt, nullopt, Time(10, 40));
    EXPECT_EQ(testee7.transcript(), "t0");
    EXPECT_EQ(testee7.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_FALSE(testee7.isComplete());
    EXPECT_EQ(testee7.alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(testee7.weekDay(), nullopt);
    EXPECT_EQ(testee7.date(), nullopt);
    EXPECT_EQ(testee7.time(), Time(10, 40));

    AddAlarmCommand testee8("t0", AlarmType::WEEKLY, 1, nullopt, Time(10, 40));
    EXPECT_EQ(testee8.transcript(), "t0");
    EXPECT_EQ(testee8.type(), CommandType::get<AddAlarmCommand>());
    EXPECT_TRUE(testee8.isComplete());
    EXPECT_EQ(testee8.alarmType(), AlarmType::WEEKLY);
    EXPECT_EQ(testee8.weekDay(), 1);
    EXPECT_EQ(testee8.date(), nullopt);
    EXPECT_EQ(testee8.time(), Time(10, 40));
}

TEST(CommandTests, listAlarmsCommand_constructor_shouldSetAttributes)
{
    ListAlarmsCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListAlarmsCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, removeAlarmCommand_constructor_shouldSetAttributes)
{
    RemoveAlarmCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<RemoveAlarmCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.id(), nullopt);

    RemoveAlarmCommand testee1("t0", 10);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<RemoveAlarmCommand>());
    EXPECT_TRUE(testee1.isComplete());
    EXPECT_EQ(testee1.id(), 10);
}

TEST(ReminderManagerTests, faceDescriptorConstructor_shouldSetAttributes)
{
    FaceDescriptor testee({1.f, 2.f, 3.f});
    EXPECT_EQ(testee.data(), vector<float>({1.f, 2.f, 3.f}));
}

TEST(ReminderManagerTests, faceDescriptorDistance_notCompatible_shouldReturnInfinity)
{
    EXPECT_FLOAT_EQ(FaceDescriptor({1.f}).distance(FaceDescriptor({-1.f, 2.f})), numeric_limits<float>::infinity());
}

TEST(ReminderManagerTests, faceDescriptorDistance_shouldReturnDistance)
{
    EXPECT_FLOAT_EQ(FaceDescriptor({1.f, -2.f}).distance(FaceDescriptor({4.f, 2.f})), 5.f);
}

TEST(ReminderManagerTests, faceDescriptorMean_empty_shouldReturnEmpty)
{
    auto testee = FaceDescriptor::mean({});
    EXPECT_EQ(testee.data(), vector<float>({}));
}

TEST(ReminderManagerTests, faceDescriptorMean_one_shouldReturnTheElement)
{
    auto testee = FaceDescriptor::mean({FaceDescriptor({1.f, 3.f})});
    EXPECT_EQ(testee.data(), vector<float>({1.f, 3.f}));
}

TEST(ReminderManagerTests, faceDescriptorMean_many_shouldReturnTheMean)
{
    auto testee = FaceDescriptor::mean({FaceDescriptor({1.f, 3.f, 5.f}), FaceDescriptor({3.f, 9.f, -5.f})});
    EXPECT_EQ(testee.data(), vector<float>({2.f, 6.f, 0.f}));
}

TEST(CommandTests, addReminderCommand_constructor_shouldSetAttributes)
{
    AddReminderCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<AddReminderCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.text(), nullopt);
    EXPECT_EQ(testee0.date(), nullopt);
    EXPECT_EQ(testee0.time(), nullopt);
    EXPECT_EQ(testee0.faceDescriptor(), nullopt);

    AddReminderCommand testee1("t0", "Hello", nullopt, nullopt, nullopt);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<AddReminderCommand>());
    EXPECT_FALSE(testee1.isComplete());
    EXPECT_EQ(testee1.text(), "Hello");
    EXPECT_EQ(testee1.date(), nullopt);
    EXPECT_EQ(testee1.time(), nullopt);
    EXPECT_EQ(testee1.faceDescriptor(), nullopt);

    AddReminderCommand testee2("t0", "Hello", Date(2022, 7, 3), nullopt, nullopt);
    EXPECT_EQ(testee2.transcript(), "t0");
    EXPECT_EQ(testee2.type(), CommandType::get<AddReminderCommand>());
    EXPECT_FALSE(testee2.isComplete());
    EXPECT_EQ(testee2.text(), "Hello");
    EXPECT_EQ(testee2.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee2.time(), nullopt);
    EXPECT_EQ(testee2.faceDescriptor(), nullopt);

    AddReminderCommand testee3("t0", "Hello", Date(2022, 7, 3), Time(10, 40), nullopt);
    EXPECT_EQ(testee3.transcript(), "t0");
    EXPECT_EQ(testee3.type(), CommandType::get<AddReminderCommand>());
    EXPECT_FALSE(testee3.isComplete());
    EXPECT_EQ(testee3.text(), "Hello");
    EXPECT_EQ(testee3.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee3.time(), Time(10, 40));
    EXPECT_EQ(testee3.faceDescriptor(), nullopt);

    AddReminderCommand testee4("t0", "Hello", Date(2022, 7, 3), Time(10, 40), FaceDescriptor({1.f}));
    EXPECT_EQ(testee4.transcript(), "t0");
    EXPECT_EQ(testee4.type(), CommandType::get<AddReminderCommand>());
    EXPECT_TRUE(testee4.isComplete());
    EXPECT_EQ(testee4.text(), "Hello");
    EXPECT_EQ(testee4.date(), Date(2022, 7, 3));
    EXPECT_EQ(testee4.time(), Time(10, 40));
    ASSERT_TRUE(testee4.faceDescriptor().has_value());
    EXPECT_EQ(testee4.faceDescriptor().value().data(), vector<float>({1.f}));
}

TEST(CommandTests, listRemindersCommand_constructor_shouldSetAttributes)
{
    ListRemindersCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListRemindersCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, removeReminderCommand_constructor_shouldSetAttributes)
{
    RemoveReminderCommand testee0("t0");
    EXPECT_EQ(testee0.transcript(), "t0");
    EXPECT_EQ(testee0.type(), CommandType::get<RemoveReminderCommand>());
    EXPECT_FALSE(testee0.isComplete());
    EXPECT_EQ(testee0.id(), nullopt);

    RemoveReminderCommand testee1("t0", 10);
    EXPECT_EQ(testee1.transcript(), "t0");
    EXPECT_EQ(testee1.type(), CommandType::get<RemoveReminderCommand>());
    EXPECT_TRUE(testee1.isComplete());
    EXPECT_EQ(testee1.id(), 10);
}

TEST(CommandTests, listCommandsCommand_constructor_shouldSetAttributes)
{
    ListCommandsCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<ListCommandsCommand>());
    EXPECT_TRUE(testee.isComplete());
}

TEST(CommandTests, nothingCommand_constructor_shouldSetAttributes)
{
    NothingCommand testee("t0");
    EXPECT_EQ(testee.transcript(), "t0");
    EXPECT_EQ(testee.type(), CommandType::get<NothingCommand>());
    EXPECT_TRUE(testee.isComplete());
}
