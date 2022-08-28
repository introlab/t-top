#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

#define EXPECT_ONE_COMMAND(commands, commandClassName)                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        auto commandsVar = commands;                                                                                   \
        ASSERT_EQ(commandsVar.size(), 1);                                                                              \
        EXPECT_EQ(commandsVar[0]->type(), CommandType::get<commandClassName>());                                       \
    } while (false)

class FormatterFrenchTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadFrenchStringResources(); }
};

class FormatterEnglishTests : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { loadEnglishStringResources(); }
};

TEST(FormatterTests, format_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    Formatter::clear();
    EXPECT_THROW(Formatter::format("Hello {}", 10), runtime_error);
}

TEST(FormatterTests, language_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    Formatter::clear();
    EXPECT_THROW(Formatter::language(), runtime_error);
}

TEST(FormatterTests, weekDayNames_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    Formatter::clear();
    EXPECT_THROW(Formatter::weekDayNames(), runtime_error);
}

TEST_F(FormatterFrenchTests, weekDayNames_shouldReturnAllWeekDayNames)
{
    EXPECT_EQ(
        Formatter::weekDayNames(),
        vector<string>({"dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"}));
}

TEST_F(FormatterFrenchTests, monthNames_shouldReturnAllMonthNames)
{
    EXPECT_EQ(
        Formatter::monthNames(),
        vector<string>(
            {"janvier",
             "février",
             "mars",
             "avril",
             "mai",
             "juin",
             "juillet",
             "août",
             "septembre",
             "octobre",
             "novembre",
             "décembre"}));
}

TEST_F(FormatterFrenchTests, format_int_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("hello {}", 10), "hello 10");
    EXPECT_EQ(Formatter::format("hello {name:05d}", fmt::arg("name", 42)), "hello 00042");
    EXPECT_EQ(
        Formatter::format("this is a big number: {number}", fmt::arg("number", 42000)),
        "this is a big number: 42000");
}

TEST_F(FormatterFrenchTests, format_float_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("hello {}", 50.5), "hello 50.5");
    EXPECT_EQ(Formatter::format("hello {:L}", 50.5), "hello 50,5");
    EXPECT_EQ(Formatter::format("hello {:.1Lf}", 50.52), "hello 50,5");
    EXPECT_EQ(Formatter::format("hello {:.0f}", 50.52), "hello 51");
}

TEST_F(FormatterFrenchTests, format_time_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("Il est {}.", Time(01, 05)), "Il est 1:05.");
    EXPECT_EQ(Formatter::format("Il est {}.", Time(10, 15)), "Il est 10:15.");
    EXPECT_EQ(Formatter::format("Il est {}.", Time(11, 59)), "Il est 11:59.");
    EXPECT_EQ(Formatter::format("Il est {}.", Time(12, 55)), "Il est 12:55.");
    EXPECT_EQ(Formatter::format("Il est {}.", Time(13, 00)), "Il est 13:00.");
    EXPECT_EQ(Formatter::format("Il est {}.", Time(22, 35)), "Il est 22:35.");
}

TEST_F(FormatterFrenchTests, format_date_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2000, 0, 1)), "Il est le 1 janvier 2000.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2001, 1, 10)), "Il est le 10 février 2001.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2002, 2, 20)), "Il est le 20 mars 2002.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2003, 3, 30)), "Il est le 30 avril 2003.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2004, 4, 2)), "Il est le 2 mai 2004.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2005, 5, 12)), "Il est le 12 juin 2005.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2006, 6, 22)), "Il est le 22 juillet 2006.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2007, 7, 3)), "Il est le 3 août 2007.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2008, 8, 13)), "Il est le 13 septembre 2008.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2009, 9, 23)), "Il est le 23 octobre 2009.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2010, 10, 4)), "Il est le 4 novembre 2010.");
    EXPECT_EQ(Formatter::format("Il est le {}.", Date(2011, 11, 14)), "Il est le 14 décembre 2011.");
}


TEST_F(FormatterEnglishTests, weekDayNames_shouldReturnAllWeekDayNames)
{
    EXPECT_EQ(
        Formatter::weekDayNames(),
        vector<string>({"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}));
}

TEST_F(FormatterEnglishTests, monthNames_shouldReturnAllMonthNames)
{
    EXPECT_EQ(
        Formatter::monthNames(),
        vector<string>(
            {"January",
             "February",
             "March",
             "April",
             "May",
             "June",
             "July",
             "August",
             "September",
             "October",
             "November",
             "December"}));
}

TEST_F(FormatterEnglishTests, format_int_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("hello {}", 10), "hello 10");
    EXPECT_EQ(Formatter::format("hello {name:05d}", fmt::arg("name", 42)), "hello 00042");
    EXPECT_EQ(
        Formatter::format("this is a big number: {number}", fmt::arg("number", 42000)),
        "this is a big number: 42000");
}

TEST_F(FormatterEnglishTests, format_float_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("hello {}", 50.5), "hello 50.5");
    EXPECT_EQ(Formatter::format("hello {:L}", 50.5), "hello 50.5");
    EXPECT_EQ(Formatter::format("hello {:.1Lf}", 50.52), "hello 50.5");
    EXPECT_EQ(Formatter::format("hello {:.0f}", 50.52), "hello 51");
}

TEST_F(FormatterEnglishTests, format_time_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("It's {}.", Time(01, 05)), "It's 1:05 AM.");
    EXPECT_EQ(Formatter::format("It's {}.", Time(10, 15)), "It's 10:15 AM.");
    EXPECT_EQ(Formatter::format("It's {}.", Time(11, 59)), "It's 11:59 AM.");
    EXPECT_EQ(Formatter::format("It's {}.", Time(12, 55)), "It's 12:55 PM.");
    EXPECT_EQ(Formatter::format("It's {}.", Time(13, 00)), "It's 1:00 PM.");
    EXPECT_EQ(Formatter::format("It's {}.", Time(22, 35)), "It's 10:35 PM.");
}

TEST_F(FormatterEnglishTests, format_date_shouldReturnOneWeatherCommand)
{
    EXPECT_EQ(Formatter::format("Date: {}", Date(2000, 0, 1)), "Date: January 1, 2000");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2001, 1, 10)), "Date: February 10, 2001");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2002, 2, 20)), "Date: March 20, 2002");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2003, 3, 30)), "Date: April 30, 2003");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2004, 4, 2)), "Date: May 2, 2004");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2005, 5, 12)), "Date: June 12, 2005");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2006, 6, 22)), "Date: July 22, 2006");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2007, 7, 3)), "Date: August 3, 2007");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2008, 8, 13)), "Date: September 13, 2008");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2009, 9, 23)), "Date: October 23, 2009");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2010, 10, 4)), "Date: November 4, 2010");
    EXPECT_EQ(Formatter::format("Date: {}", Date(2011, 11, 14)), "Date: December 14, 2011");
}
