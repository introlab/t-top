#include <home_logger_common/language/StringResources.h>

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

using namespace std;
namespace fs = boost::filesystem;

TEST(StringResourcesTests, getValue_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    EXPECT_THROW(StringResources::getValue("key6"), runtime_error);
}

TEST(StringResourcesTests, getVector_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    EXPECT_THROW(StringResources::getVector("key6"), runtime_error);
}

TEST(StringResourcesTests, language_notInitialized_shouldThrowRuntimeError)
{
    StringResources::clear();
    EXPECT_THROW(StringResources::language(), runtime_error);
}

TEST(StringResourcesTests, loadFromMap_shouldSetTheProperties)
{
    StringResources::clear();
    StringResources::loadFromMap({{"k0", "v0"}, {"k1", "[v1, v2]"}}, Language::ENGLISH);

    EXPECT_EQ(StringResources::getValue("k0"), "v0");
    EXPECT_EQ(StringResources::getVector("k1"), vector<string>({"v1", "v2"}));
    EXPECT_EQ(StringResources::language(), Language::ENGLISH);
}

TEST(StringResourcesTests, loadFromFile_shouldSetTheProperties)
{
    fs::path testFilePath(__FILE__);
    fs::path propertiesFilePath =
        testFilePath.parent_path().parent_path() / "resources" / "StringResourcesTests" / "french.properties";

    StringResources::clear();
    StringResources::loadFromFile(propertiesFilePath.string(), Language::FRENCH);

    EXPECT_EQ(StringResources::getValue("k0"), "v0");
    EXPECT_EQ(StringResources::getVector("k1"), vector<string>({"v1", "v2"}));
    EXPECT_EQ(StringResources::language(), Language::FRENCH);
}
