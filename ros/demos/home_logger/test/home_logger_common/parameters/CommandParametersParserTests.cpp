#include <home_logger_common/parameters/CommandParametersParser.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

class CommandParametersParserFrenchTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadFrenchStringResources(); }
};

class CommandParametersParserEnglishTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadEnglishStringResources(); }
};

TEST(CommandParametersParserTests, findInt_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findInt(""), nullopt);
    EXPECT_EQ(findInt("Bonjour"), nullopt);
}

TEST(CommandParametersParserTests, findInt_valid_shouldReturnTheNumber)
{
    EXPECT_EQ(findInt("1"), 1);
    EXPECT_EQ(findInt(" 23 "), 23);
    EXPECT_EQ(findInt("Alarme 456 enlever"), 456);
    EXPECT_EQ(findInt("est-ce que tu peux enlever l'alarme 1"), 1);
    EXPECT_EQ(findInt("780 10"), 780);
    EXPECT_EQ(findInt("9"), 9);
    EXPECT_EQ(findInt("-9"), -9);
    EXPECT_EQ(findInt("1-"), 1);
}

TEST_F(CommandParametersParserFrenchTests, findTime_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findTime(""), nullopt);
    EXPECT_EQ(findTime("10"), nullopt);
    EXPECT_EQ(findTime("10 00"), nullopt);

    EXPECT_EQ(findTime("allo :"), nullopt);
    EXPECT_EQ(findTime("allo h"), nullopt);
    EXPECT_EQ(findTime("allo h"), nullopt);

    EXPECT_EQ(findTime("-1h"), nullopt);
    EXPECT_EQ(findTime("24 h"), nullopt);

    EXPECT_EQ(findTime("1h-1"), nullopt);
    EXPECT_EQ(findTime("1h60"), nullopt);

    EXPECT_EQ(findTime("10h00 AM PM"), nullopt);
    EXPECT_EQ(findTime("12h00 AM"), nullopt);
}

TEST_F(CommandParametersParserFrenchTests, findTime_valid_shouldReturnTheTime)
{
    EXPECT_EQ(findTime("10h"), Time(10, 0));
    EXPECT_EQ(findTime("10h15"), Time(10, 15));
    EXPECT_EQ(findTime("7h 00 PM"), Time(19, 0));
    EXPECT_EQ(findTime("7 h00 en après-midi"), Time(19, 0));
    EXPECT_EQ(findTime("7 h 00 en après midi"), Time(19, 0));
    EXPECT_EQ(findTime("7h00 p.m."), Time(19, 0));
}

TEST_F(CommandParametersParserFrenchTests, findDate_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findDate("", 0, 0), nullopt);
    EXPECT_EQ(findDate("0", 0, 0), nullopt);
    EXPECT_EQ(findDate("32", 0, 0), nullopt);

    EXPECT_EQ(findDate("janvier", 0, 0), nullopt);
    EXPECT_EQ(findDate("0 février", 0, 0), nullopt);
    EXPECT_EQ(findDate("32 février", 0, 0), nullopt);
    EXPECT_EQ(findDate("février 2022", 0, 0), nullopt);

    EXPECT_EQ(findDate("1 janvier février", 0, 0), nullopt);
}

TEST_F(CommandParametersParserFrenchTests, findDate_valid_shouldReturnTheDate)
{
    EXPECT_TRUE(findDate("aujourd'hui", 2022, 0).has_value());

    EXPECT_EQ(findDate("1", 2022, 0), Date(2022, 0, 1));

    EXPECT_EQ(findDate("1 janvier", 2022, 0), Date(2022, 0, 1));
    EXPECT_EQ(findDate("2 février", 2022, 0), Date(2022, 1, 2));
    EXPECT_EQ(findDate("3 mars", 2022, 0), Date(2022, 2, 3));
    EXPECT_EQ(findDate("4 avril", 2022, 0), Date(2022, 3, 4));
    EXPECT_EQ(findDate("5 mai", 2022, 0), Date(2022, 4, 5));
    EXPECT_EQ(findDate("6 juin", 2022, 0), Date(2022, 5, 6));
    EXPECT_EQ(findDate("7 juillet", 2022, 0), Date(2022, 6, 7));
    EXPECT_EQ(findDate("8 août", 2022, 0), Date(2022, 7, 8));
    EXPECT_EQ(findDate("9 septembre", 2022, 0), Date(2022, 8, 9));
    EXPECT_EQ(findDate("10 octobre", 2022, 0), Date(2022, 9, 10));
    EXPECT_EQ(findDate("11 novembre", 2022, 0), Date(2022, 10, 11));
    EXPECT_EQ(findDate("12 décembre", 2022, 0), Date(2022, 11, 12));

    EXPECT_EQ(findDate("12 décembre 2023", 2022, 0), Date(2023, 11, 12));
}

TEST_F(CommandParametersParserFrenchTests, findWeekDay_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findWeekDay(""), nullopt);
    EXPECT_EQ(findWeekDay("1"), nullopt);
    EXPECT_EQ(findWeekDay("mar"), nullopt);
    EXPECT_EQ(findWeekDay("dimanche lundi"), nullopt);
}

TEST_F(CommandParametersParserFrenchTests, findWeekDay_valid_shouldReturnTheWeekDay)
{
    EXPECT_EQ(findWeekDay("dimanche  "), 0);
    EXPECT_EQ(findWeekDay(" lundi  "), 1);
    EXPECT_EQ(findWeekDay(" mardi  "), 2);
    EXPECT_EQ(findWeekDay("mercredi"), 3);
    EXPECT_EQ(findWeekDay("jeudI"), 4);
    EXPECT_EQ(findWeekDay("venDredi"), 5);
    EXPECT_EQ(findWeekDay("Samedi"), 6);
}


TEST_F(CommandParametersParserEnglishTests, findTime_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findTime(""), nullopt);
    EXPECT_EQ(findTime("10"), nullopt);
    EXPECT_EQ(findTime("10 00"), nullopt);

    EXPECT_EQ(findTime("allo :"), nullopt);
    EXPECT_EQ(findTime("allo h"), nullopt);
    EXPECT_EQ(findTime("allo h"), nullopt);

    EXPECT_EQ(findTime("-1h"), nullopt);
    EXPECT_EQ(findTime("24 h"), nullopt);

    EXPECT_EQ(findTime("1h-1"), nullopt);
    EXPECT_EQ(findTime("1h60"), nullopt);

    EXPECT_EQ(findTime("10 AM PM"), nullopt);
    EXPECT_EQ(findTime("10h00 AM PM"), nullopt);
}

TEST_F(CommandParametersParserEnglishTests, findTime_valid_shouldReturnTheTime)
{
    EXPECT_EQ(findTime("10 hour"), Time(10, 0));
    EXPECT_EQ(findTime("10:15"), Time(10, 15));
    EXPECT_EQ(findTime("7:00 PM"), Time(19, 0));
    EXPECT_EQ(findTime("7:00 en afternoon"), Time(19, 0));
    EXPECT_EQ(findTime("7:00 p.m."), Time(19, 0));
    EXPECT_EQ(findTime("8 p.m."), Time(20, 0));
    EXPECT_EQ(findTime("10 AM"), Time(10, 0));
}

TEST_F(CommandParametersParserEnglishTests, findDate_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findDate("", 0, 0), nullopt);
    EXPECT_EQ(findDate("0", 0, 0), nullopt);
    EXPECT_EQ(findDate("32", 0, 0), nullopt);
    EXPECT_EQ(findDate("32 31", 0, 0), nullopt);

    EXPECT_EQ(findDate("January", 0, 0), nullopt);
    EXPECT_EQ(findDate("february 0", 0, 0), nullopt);
    EXPECT_EQ(findDate("february 32", 0, 0), nullopt);
    EXPECT_EQ(findDate("february 2022", 0, 0), nullopt);
}

TEST_F(CommandParametersParserEnglishTests, findDate_valid_shouldReturnTheDate)
{
    EXPECT_TRUE(findDate("today", 2022, 0).has_value());

    EXPECT_EQ(findDate("1", 2022, 0), Date(2022, 0, 1));

    EXPECT_EQ(findDate("January 1st", 2022, 0), Date(2022, 0, 1));
    EXPECT_EQ(findDate("February 2nd", 2022, 0), Date(2022, 1, 2));
    EXPECT_EQ(findDate("March 3th", 2022, 0), Date(2022, 2, 3));
    EXPECT_EQ(findDate("April 4", 2022, 0), Date(2022, 3, 4));
    EXPECT_EQ(findDate("May 5", 2022, 0), Date(2022, 4, 5));
    EXPECT_EQ(findDate("June 6", 2022, 0), Date(2022, 5, 6));
    EXPECT_EQ(findDate("July 7th", 2022, 0), Date(2022, 6, 7));
    EXPECT_EQ(findDate("August 8", 2022, 0), Date(2022, 7, 8));
    EXPECT_EQ(findDate("September 9", 2022, 0), Date(2022, 8, 9));
    EXPECT_EQ(findDate("October 10", 2022, 0), Date(2022, 9, 10));
    EXPECT_EQ(findDate("November 11", 2022, 0), Date(2022, 10, 11));
    EXPECT_EQ(findDate("December 12", 2022, 0), Date(2022, 11, 12));

    EXPECT_EQ(findDate("December 12, 2023", 2022, 0), Date(2023, 11, 12));
}

TEST_F(CommandParametersParserEnglishTests, findWeekDay_invalid_shouldReturnNullopt)
{
    EXPECT_EQ(findWeekDay(""), nullopt);
    EXPECT_EQ(findWeekDay("1"), nullopt);
    EXPECT_EQ(findWeekDay("mar"), nullopt);
    EXPECT_EQ(findWeekDay("Sunday Monday"), nullopt);
}

TEST_F(CommandParametersParserEnglishTests, findWeekDay_valid_shouldReturnTheWeekDay)
{
    EXPECT_EQ(findWeekDay("sunday  "), 0);
    EXPECT_EQ(findWeekDay(" Monday  "), 1);
    EXPECT_EQ(findWeekDay(" tuesday  "), 2);
    EXPECT_EQ(findWeekDay("Wednesday"), 3);
    EXPECT_EQ(findWeekDay("Thursday"), 4);
    EXPECT_EQ(findWeekDay("Friday"), 5);
    EXPECT_EQ(findWeekDay("Saturday"), 6);
}
