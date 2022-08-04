#include <home_logger_common/Properties.h>

#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

using namespace std;
namespace fs = boost::filesystem;

TEST(PropertiesTests, constructor_map_shouldCopyTheMap)
{
    Properties properties({{"key0", "v1"}, {"key1", "v2"}});

    EXPECT_EQ(properties.get<string>("key0"), "v1");
    EXPECT_EQ(properties.get<string>("key1"), "v2");
    EXPECT_EQ(properties.keys(), unordered_set<string>({"key0", "key1"}));
}

TEST(PropertiesTests, constructor_file_shouldReadTheProperties)
{
    fs::path testFilePath(__FILE__);
    fs::path propertiesFilePath = testFilePath.parent_path() / "resources" / "PropertiesTests" / "valid.properties";
    Properties properties(propertiesFilePath.string());

    EXPECT_EQ(properties.get<string>("key0"), "abc");
    EXPECT_EQ(properties.get<string>("key1"), "abc");
    EXPECT_EQ(properties.get<string>("key3"), "abc");

    EXPECT_EQ(properties.get<string>("key4"), "");

    EXPECT_THROW(properties.get<string>("key5"), runtime_error);

    EXPECT_THROW(properties.get<int>("key4"), runtime_error);

    EXPECT_EQ(properties.get<int>("key_array[0]"), 10);
    EXPECT_EQ(properties.get<int>("key_array[1]"), 10);

    EXPECT_EQ(properties.get<double>("key_array[1]"), 10.5);

    EXPECT_THROW(properties.get<vector<int>>("key6"), runtime_error);
    EXPECT_THROW(properties.get<vector<int>>("key7"), runtime_error);
    EXPECT_THROW(properties.get<vector<int>>("key8"), runtime_error);
    EXPECT_EQ(properties.get<vector<int>>("key9"), vector<int>());
    EXPECT_EQ(properties.get<vector<int>>("key10"), vector<int>({1, 2, 3, 4}));

    EXPECT_EQ(properties.get<bool>("key11"), true);
    EXPECT_EQ(properties.get<bool>("key12"), true);
    EXPECT_EQ(properties.get<bool>("key13"), false);
}
