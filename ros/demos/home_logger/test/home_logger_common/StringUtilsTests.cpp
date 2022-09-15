#include <home_logger_common/StringUtils.h>

#include <gtest/gtest.h>

using namespace std;

TEST(StringUtilsTests, toUpperString_should)
{
    EXPECT_EQ(toUpperString("wEAther"), "WEATHER");
    EXPECT_EQ(toUpperString("méTÉo"), "MÉTÉO");
}

TEST(StringUtilsTests, toLowerString_should)
{
    EXPECT_EQ(toLowerString("wEAther"), "weather");
    EXPECT_EQ(toLowerString("méTÉo"), "météo");
}

TEST(StringUtilsTests, trimLeft_should)
{
    string teste = "  \t\nwEAther\t\t\n ";
    EXPECT_EQ(trimLeft(teste), "wEAther\t\t\n ");
    EXPECT_EQ(teste, "wEAther\t\t\n ");
}

TEST(StringUtilsTests, trimRight_should)
{
    string teste = "  \t\nwEAther\t\t\n ";
    EXPECT_EQ(trimRight(teste), "  \t\nwEAther");
    EXPECT_EQ(teste, "  \t\nwEAther");
}

TEST(StringUtilsTests, trim_should)
{
    string teste = "  \t\nwEAther\t\t\n ";
    EXPECT_EQ(trim(teste), "wEAther");
    EXPECT_EQ(teste, "wEAther");
}
