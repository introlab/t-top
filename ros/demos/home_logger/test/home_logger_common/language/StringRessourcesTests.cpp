#include <home_logger_common/language/StringRessources.h>

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

using namespace std;
namespace fs = boost::filesystem;

TEST(StringRessourcesTests, getValue_notInitialized_shouldThrowRuntimeError)
{
    StringRessources::clear();
    EXPECT_THROW(StringRessources::getValue("key6"), runtime_error);
}

TEST(StringRessourcesTests, getVector_notInitialized_shouldThrowRuntimeError)
{
    StringRessources::clear();
    EXPECT_THROW(StringRessources::getVector("key6"), runtime_error);
}

TEST(StringRessourcesTests, language_notInitialized_shouldThrowRuntimeError)
{
    StringRessources::clear();
    EXPECT_THROW(StringRessources::language(), runtime_error);
}

TEST(StringRessourcesTests, loadFromMap_shouldSetTheProperties)
{
    StringRessources::clear();
    StringRessources::loadFromMap({{"k0", "v0"}, {"k1", "[v1, v2]"}}, Language::ENGLISH);

    EXPECT_EQ(StringRessources::getValue("k0"), "v0");
    EXPECT_EQ(StringRessources::getVector("k1"), vector<string>({"v1", "v2"}));
    EXPECT_EQ(StringRessources::language(), Language::ENGLISH);
}

TEST(StringRessourcesTests, loadFromFile_shouldSetTheProperties)
{
    fs::path testFilePath(__FILE__);
    fs::path propertiesFilePath =
        testFilePath.parent_path().parent_path() / "resources" / "StringRessourcesTests" / "french.properties";


    StringRessources::clear();
    StringRessources::loadFromFile(propertiesFilePath.string(), Language::FRENCH);

    EXPECT_EQ(StringRessources::getValue("k0"), "v0");
    EXPECT_EQ(StringRessources::getVector("k1"), vector<string>({"v1", "v2"}));
    EXPECT_EQ(StringRessources::language(), Language::FRENCH);
}
