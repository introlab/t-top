#include <home_logger_common/DateTime.h>

#include <gtest/gtest.h>

using namespace std;

TEST(DateTimeTests, timeConstructor_default_shouldSetAllAttributesTo0)
{
    Time testee;
    EXPECT_EQ(testee.hour, 0);
    EXPECT_EQ(testee.minute, 0);
}

TEST(DateTimeTests, timeConstructor_values_shouldSetAllAttributes)
{
    Time testee(10, 30);
    EXPECT_EQ(testee.hour, 10);
    EXPECT_EQ(testee.minute, 30);
}

TEST(DateTimeTests, timeNow_shouldSetAllAttributes)
{
    Time testee = Time::now();
    EXPECT_GE(testee.hour, 0);
    EXPECT_GE(testee.minute, 0);
}

TEST(DateTimeTests, timeBetween_shouldReturnTheRightValue)
{
    EXPECT_TRUE(Time(13, 30).between(Time(13, 30), Time(14, 00)));
    EXPECT_TRUE(Time(13, 30).between(Time(13, 00), Time(13, 30)));
    EXPECT_TRUE(Time(13, 30).between(Time(13, 00), Time(14, 00)));
    EXPECT_FALSE(Time(12, 30).between(Time(13, 00), Time(14, 00)));
    EXPECT_FALSE(Time(14, 30).between(Time(13, 00), Time(14, 00)));

    EXPECT_TRUE(Time(22, 30).between(Time(22, 30), Time(7, 00)));
    EXPECT_TRUE(Time(7, 00).between(Time(22, 30), Time(7, 00)));
    EXPECT_TRUE(Time(23, 00).between(Time(22, 30), Time(7, 00)));
    EXPECT_TRUE(Time(6, 00).between(Time(22, 30), Time(7, 00)));
    EXPECT_FALSE(Time(7, 30).between(Time(22, 30), Time(7, 00)));
    EXPECT_FALSE(Time(22, 00).between(Time(22, 30), Time(7, 00)));
}

TEST(DateTimeTests, timeComparisonOperators_shouldReturnTheRightValue)
{
    EXPECT_TRUE(Time(10, 30) == Time(10, 30));
    EXPECT_FALSE(Time(10, 30) == Time(10, 0));
    EXPECT_FALSE(Time(10, 30) == Time(2, 30));

    EXPECT_TRUE(Time(2, 15) != Time(3, 15));
    EXPECT_TRUE(Time(2, 15) != Time(2, 10));
    EXPECT_FALSE(Time(5, 45) != Time(5, 45));

    EXPECT_TRUE(Time(2, 15) < Time(3, 15));
    EXPECT_TRUE(Time(2, 15) < Time(2, 16));
    EXPECT_FALSE(Time(2, 15) < Time(2, 15));
    EXPECT_FALSE(Time(2, 15) < Time(2, 14));
    EXPECT_FALSE(Time(2, 15) < Time(1, 50));

    EXPECT_TRUE(Time(2, 15) <= Time(3, 15));
    EXPECT_TRUE(Time(2, 15) <= Time(2, 16));
    EXPECT_TRUE(Time(2, 15) <= Time(2, 15));
    EXPECT_FALSE(Time(2, 15) <= Time(2, 14));
    EXPECT_FALSE(Time(2, 15) <= Time(1, 50));

    EXPECT_TRUE(Time(2, 15) > Time(2, 14));
    EXPECT_TRUE(Time(2, 15) > Time(1, 50));
    EXPECT_FALSE(Time(2, 15) > Time(2, 15));
    EXPECT_FALSE(Time(2, 15) > Time(3, 15));
    EXPECT_FALSE(Time(2, 15) > Time(2, 16));

    EXPECT_TRUE(Time(2, 15) >= Time(2, 14));
    EXPECT_TRUE(Time(2, 15) >= Time(1, 50));
    EXPECT_TRUE(Time(2, 15) >= Time(2, 15));
    EXPECT_FALSE(Time(2, 15) >= Time(3, 15));
    EXPECT_FALSE(Time(2, 15) >= Time(2, 16));
}


TEST(DateTimeTests, dateConstructor_default_shouldSetAllAttributesTo0)
{
    Date testee;
    EXPECT_EQ(testee.year, 0);
    EXPECT_EQ(testee.month, 0);
    EXPECT_EQ(testee.day, 0);
}

TEST(DateTimeTests, dateConstructor_values_shouldSetAllAttributes)
{
    Date testee(2022, 10, 30);
    EXPECT_EQ(testee.year, 2022);
    EXPECT_EQ(testee.month, 10);
    EXPECT_EQ(testee.day, 30);
}

TEST(DateTimeTests, dateNow_shouldSetAllAttributes)
{
    Date testee = Date::now();
    EXPECT_GE(testee.year, 2022);
    EXPECT_GE(testee.month, 0);
    EXPECT_GE(testee.day, 1);
}

TEST(DateTimeTests, dateWeekDay_shouldReturnTheWeekDay)
{
    Date testee(2022, 7, 3);
    EXPECT_EQ(testee.weekDay(), 3);
}

TEST(DateTimeTests, dateComparisonOperators_shouldReturnTheRightValue)
{
    EXPECT_TRUE(Date(2022, 10, 30) == Date(2022, 10, 30));
    EXPECT_FALSE(Date(2022, 10, 30) == Date(2022, 10, 0));
    EXPECT_FALSE(Date(2022, 10, 30) == Date(2022, 2, 30));
    EXPECT_FALSE(Date(2022, 10, 30) == Date(2021, 10, 30));

    EXPECT_TRUE(Date(2022, 2, 15) != Date(2021, 2, 15));
    EXPECT_TRUE(Date(2022, 2, 15) != Date(2022, 3, 15));
    EXPECT_TRUE(Date(2022, 2, 15) != Date(2022, 2, 10));
    EXPECT_FALSE(Date(2022, 5, 15) != Date(2022, 5, 15));

    EXPECT_TRUE(Date(2021, 2, 15) < Date(2022, 2, 15));
    EXPECT_TRUE(Date(2022, 2, 15) < Date(2022, 3, 15));
    EXPECT_TRUE(Date(2022, 2, 15) < Date(2022, 2, 16));
    EXPECT_FALSE(Date(2022, 2, 15) < Date(2021, 2, 15));
    EXPECT_FALSE(Date(2022, 2, 15) < Date(2021, 1, 15));
    EXPECT_FALSE(Date(2022, 2, 15) < Date(2022, 2, 14));

    EXPECT_TRUE(Date(2021, 2, 15) <= Date(2022, 2, 15));
    EXPECT_TRUE(Date(2022, 2, 15) <= Date(2022, 3, 15));
    EXPECT_TRUE(Date(2022, 2, 15) <= Date(2022, 2, 16));
    EXPECT_TRUE(Date(2022, 2, 15) <= Date(2022, 2, 15));
    EXPECT_FALSE(Date(2022, 2, 15) <= Date(2021, 1, 15));
    EXPECT_FALSE(Date(2022, 2, 15) <= Date(2022, 2, 14));

    EXPECT_TRUE(Date(2022, 2, 15) > Date(2021, 1, 15));
    EXPECT_TRUE(Date(2022, 2, 15) > Date(2022, 2, 14));
    EXPECT_FALSE(Date(2021, 2, 15) > Date(2022, 2, 15));
    EXPECT_FALSE(Date(2022, 2, 15) > Date(2022, 3, 15));
    EXPECT_FALSE(Date(2022, 2, 15) > Date(2022, 2, 16));
    EXPECT_FALSE(Date(2022, 2, 15) > Date(2022, 2, 15));

    EXPECT_TRUE(Date(2022, 2, 15) >= Date(2021, 2, 15));
    EXPECT_TRUE(Date(2022, 2, 15) >= Date(2021, 1, 15));
    EXPECT_TRUE(Date(2022, 2, 15) >= Date(2022, 2, 14));
    EXPECT_FALSE(Date(2021, 2, 15) >= Date(2022, 2, 15));
    EXPECT_FALSE(Date(2022, 2, 15) >= Date(2022, 3, 15));
    EXPECT_FALSE(Date(2022, 2, 15) >= Date(2022, 2, 16));
}


TEST(DateTimeTests, dateTimeConstructor_default_shouldSetAllAttributesTo0)
{
    DateTime testee;
    EXPECT_EQ(testee.date.year, 0);
    EXPECT_EQ(testee.date.month, 0);
    EXPECT_EQ(testee.date.day, 0);
    EXPECT_EQ(testee.time.hour, 0);
    EXPECT_EQ(testee.time.minute, 0);
}

TEST(DateTimeTests, dateTimeConstructor_values_shouldSetAllAttributes)
{
    DateTime testee(2022, 10, 30, 5, 35);
    EXPECT_EQ(testee.date.year, 2022);
    EXPECT_EQ(testee.date.month, 10);
    EXPECT_EQ(testee.date.day, 30);
    EXPECT_EQ(testee.time.hour, 5);
    EXPECT_EQ(testee.time.minute, 35);
}

TEST(DateTimeTests, dateTimeConstructor_dateTime_shouldSetAllAttributes)
{
    DateTime testee(Date(2022, 10, 30), Time(5, 35));
    EXPECT_EQ(testee.date.year, 2022);
    EXPECT_EQ(testee.date.month, 10);
    EXPECT_EQ(testee.date.day, 30);
    EXPECT_EQ(testee.time.hour, 5);
    EXPECT_EQ(testee.time.minute, 35);
}

TEST(DateTimeTests, dateTimeNow_shouldSetAllAttributes)
{
    DateTime testee = DateTime::now();
    EXPECT_GE(testee.date.year, 2022);
    EXPECT_GE(testee.date.month, 7);
    EXPECT_GE(testee.date.month, 3);
    EXPECT_GE(testee.time.hour, 0);
    EXPECT_GE(testee.time.minute, 0);
}

TEST(DateTimeTests, dateTimeComparisonOperators_shouldReturnTheRightValue)
{
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) == DateTime(Date(2022, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) == DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) == DateTime(Date(2022, 7, 3), Time(10, 31)));

    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) != DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) != DateTime(Date(2022, 7, 3), Time(10, 31)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) != DateTime(Date(2022, 7, 3), Time(10, 30)));

    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) < DateTime(Date(2022, 7, 3), Time(10, 31)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) < DateTime(Date(2022, 7, 4), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) < DateTime(Date(2022, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) < DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) < DateTime(Date(2022, 7, 3), Time(10, 29)));

    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) <= DateTime(Date(2022, 7, 3), Time(10, 31)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) <= DateTime(Date(2022, 7, 4), Time(10, 30)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) <= DateTime(Date(2022, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) <= DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) <= DateTime(Date(2022, 7, 3), Time(10, 29)));

    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) > DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) > DateTime(Date(2022, 7, 3), Time(10, 29)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) > DateTime(Date(2022, 7, 3), Time(10, 31)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) > DateTime(Date(2022, 7, 4), Time(10, 30)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) > DateTime(Date(2022, 7, 3), Time(10, 30)));

    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) >= DateTime(Date(2022, 7, 3), Time(10, 30)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) >= DateTime(Date(2021, 7, 3), Time(10, 30)));
    EXPECT_TRUE(DateTime(Date(2022, 7, 3), Time(10, 30)) >= DateTime(Date(2022, 7, 3), Time(10, 29)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) >= DateTime(Date(2022, 7, 3), Time(10, 31)));
    EXPECT_FALSE(DateTime(Date(2022, 7, 3), Time(10, 30)) >= DateTime(Date(2022, 7, 4), Time(10, 30)));
}
