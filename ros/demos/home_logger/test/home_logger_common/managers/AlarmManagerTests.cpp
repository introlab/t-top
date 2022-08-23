#include <home_logger_common/managers/AlarmManager.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

TEST(AlarmManagerTests, punctualAlarmConstructors_shouldSetAttributes)
{
    PunctualAlarm testee0(Date(2022, 8, 1), Time(22, 15));
    EXPECT_EQ(testee0.id(), tl::nullopt);
    EXPECT_EQ(testee0.date(), Date(2022, 8, 1));
    EXPECT_EQ(testee0.time(), Time(22, 15));

    PunctualAlarm testee1(10, Date(2022, 8, 2), Time(21, 15));
    EXPECT_EQ(testee1.id(), 10);
    EXPECT_EQ(testee1.date(), Date(2022, 8, 2));
    EXPECT_EQ(testee1.time(), Time(21, 15));
}

TEST(AlarmManagerTests, toSpeech_punctualFrench_shouldReturn)
{
    loadFrenchStringResources();
    PunctualAlarm testee(1, Date(2022, 8, 1), Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "L'alarme 1 est le 1 septembre 2022 à 22:15.");
}

TEST(AlarmManagerTests, toSpeech_punctualEnglish_shouldReturn)
{
    loadEnglishStringResources();
    PunctualAlarm testee(2, Date(2022, 8, 1), Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "Alarm 2 is on September 1, 2022 at 10:15 PM.");
}

TEST(AlarmManagerTests, daylyAlarmConstructors_shouldSetAttributes)
{
    DaylyAlarm testee0(Time(20, 15));
    EXPECT_EQ(testee0.id(), tl::nullopt);
    EXPECT_EQ(testee0.time(), Time(20, 15));

    DaylyAlarm testee1(11, Time(19, 15));
    EXPECT_EQ(testee1.id(), 11);
    EXPECT_EQ(testee1.time(), Time(19, 15));
}

TEST(AlarmManagerTests, toSpeech_daylyFrench_shouldReturn)
{
    loadFrenchStringResources();
    DaylyAlarm testee(3, Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "L'alarme 3 est à 22:15 à chaque jour.");
}

TEST(AlarmManagerTests, toSpeech_daylyEnglish_shouldReturn)
{
    loadEnglishStringResources();
    DaylyAlarm testee(4, Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "Alarm 4 is at 10:15 PM every day.");
}

TEST(AlarmManagerTests, weeklyAlarmConstructors_shouldSetAttributes)
{
    WeeklyAlarm testee0(2, Time(18, 15));
    EXPECT_EQ(testee0.id(), tl::nullopt);
    EXPECT_EQ(testee0.weekDay(), 2);
    EXPECT_EQ(testee0.time(), Time(18, 15));

    WeeklyAlarm testee1(12, 3, Time(17, 15));
    EXPECT_EQ(testee1.id(), 12);
    EXPECT_EQ(testee1.weekDay(), 3);
    EXPECT_EQ(testee1.time(), Time(17, 15));
}

TEST(AlarmManagerTests, toSpeech_weeklyFrench_shouldReturn)
{
    loadFrenchStringResources();
    WeeklyAlarm testee(5, 3, Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "L'alarme 5 est tous les mercredi à 22:15.");
}

TEST(AlarmManagerTests, toSpeech_weeklyEnglish_shouldReturn)
{
    loadEnglishStringResources();
    WeeklyAlarm testee(6, 4, Time(22, 15));
    EXPECT_EQ(testee.toSpeech(), "Alarm 6 is at 10:15 PM every Thursday.");
}

TEST(AlarmManagerTests, toAlarm_incomplete_shouldThrow)
{
    EXPECT_THROW(toAlarm(AddAlarmCommand("allo")), runtime_error);
}

TEST(AlarmManagerTests, toAlarm_punctual_shouldReturnPuntualAlarm)
{
    auto alarm = toAlarm(AddAlarmCommand("allo", AlarmType::PUNCTUAL, tl::nullopt, Date(2022, 8, 1), Time(6, 30)));

    auto punctualAlarm = dynamic_cast<const PunctualAlarm&>(*alarm);
    EXPECT_EQ(punctualAlarm.date(), Date(2022, 8, 1));
    EXPECT_EQ(punctualAlarm.time(), Time(6, 30));
}

TEST(AlarmManagerTests, toAlarm_dayly_shouldReturnDaylyAlarm)
{
    auto alarm = toAlarm(AddAlarmCommand("allo", AlarmType::DAYLY, tl::nullopt, tl::nullopt, Time(7, 30)));

    auto daylyAlarm = dynamic_cast<const DaylyAlarm&>(*alarm);
    EXPECT_EQ(daylyAlarm.time(), Time(7, 30));
}

TEST(AlarmManagerTests, toAlarm_weekly_shouldReturnWeeklyAlarm)
{
    auto alarm = toAlarm(AddAlarmCommand("allo", AlarmType::WEEKLY, 0, tl::nullopt, Time(8, 30)));

    auto weeklyAlarm = dynamic_cast<const WeeklyAlarm&>(*alarm);
    EXPECT_EQ(weeklyAlarm.weekDay(), 0);
    EXPECT_EQ(weeklyAlarm.time(), Time(8, 30));
}

TEST(AlarmManagerTests, insertListRemove_shouldInsertListAndRemove)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    AlarmManager testee(database);

    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    testee.insertAlarm(make_unique<DaylyAlarm>(Time(21, 15)));
    testee.insertAlarm(make_unique<WeeklyAlarm>(1, Time(6, 15)));

    auto alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 3);

    auto punctualAlarm = dynamic_cast<const PunctualAlarm&>(*alarms[0]);
    EXPECT_EQ(punctualAlarm.id(), 1);
    EXPECT_EQ(punctualAlarm.date(), Date(2022, 8, 1));
    EXPECT_EQ(punctualAlarm.time(), Time(22, 15));

    auto daylyAlarm = dynamic_cast<const DaylyAlarm&>(*alarms[1]);
    EXPECT_EQ(daylyAlarm.id(), 2);
    EXPECT_EQ(daylyAlarm.time(), Time(21, 15));

    auto weeklyAlarm = dynamic_cast<const WeeklyAlarm&>(*alarms[2]);
    EXPECT_EQ(weeklyAlarm.id(), 3);
    EXPECT_EQ(weeklyAlarm.weekDay(), 1);
    EXPECT_EQ(weeklyAlarm.time(), Time(6, 15));

    testee.removeAlarm(2);
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 2);

    punctualAlarm = dynamic_cast<const PunctualAlarm&>(*alarms[0]);
    EXPECT_EQ(punctualAlarm.id(), 1);
    EXPECT_EQ(punctualAlarm.date(), Date(2022, 8, 1));
    EXPECT_EQ(punctualAlarm.time(), Time(22, 15));

    weeklyAlarm = dynamic_cast<const WeeklyAlarm&>(*alarms[1]);
    EXPECT_EQ(weeklyAlarm.id(), 3);
    EXPECT_EQ(weeklyAlarm.weekDay(), 1);
    EXPECT_EQ(weeklyAlarm.time(), Time(6, 15));
}

TEST(AlarmManagerTests, insert_shouldReplaceIds)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    AlarmManager testee(database);

    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));

    auto alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 3);
    EXPECT_EQ(alarms[0]->id(), 1);
    EXPECT_EQ(alarms[1]->id(), 2);
    EXPECT_EQ(alarms[2]->id(), 3);

    testee.removeAlarm(1);
    testee.removeAlarm(3);
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 1);
    EXPECT_EQ(alarms[0]->id(), 2);

    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 2);
    EXPECT_EQ(alarms[0]->id(), 1);
    EXPECT_EQ(alarms[1]->id(), 2);

    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 4);
    EXPECT_EQ(alarms[0]->id(), 1);
    EXPECT_EQ(alarms[1]->id(), 2);
    EXPECT_EQ(alarms[2]->id(), 3);
    EXPECT_EQ(alarms[3]->id(), 4);

    testee.removeAlarm(2);
    testee.removeAlarm(3);
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 2);
    EXPECT_EQ(alarms[0]->id(), 1);
    EXPECT_EQ(alarms[1]->id(), 4);

    testee.insertAlarm(make_unique<PunctualAlarm>(Date(2022, 8, 1), Time(22, 15)));
    alarms = testee.listAlarms();
    ASSERT_EQ(alarms.size(), 3);
    EXPECT_EQ(alarms[0]->id(), 1);
    EXPECT_EQ(alarms[1]->id(), 2);
    EXPECT_EQ(alarms[2]->id(), 4);
}
