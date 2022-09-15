#include <home_logger_common/language/Language.h>

#include <gtest/gtest.h>

using namespace std;

TEST(LanguageTests, languageFromString_en_shouldReturnTrueAndSetEnglish)
{
    Language language = Language::FRENCH;
    EXPECT_TRUE(languageFromString("en", language));
    EXPECT_EQ(language, Language::ENGLISH);
}

TEST(LanguageTests, languageFromString_fr_shouldReturnTrueAndSetFrench)
{
    Language language = Language::ENGLISH;
    EXPECT_TRUE(languageFromString("fr", language));
    EXPECT_EQ(language, Language::FRENCH);
}

TEST(LanguageTests, languageFromString_invalid_shouldReturnFalse)
{
    Language language = Language::ENGLISH;
    EXPECT_FALSE(languageFromString("fra", language));
    EXPECT_FALSE(languageFromString("english", language));
}
