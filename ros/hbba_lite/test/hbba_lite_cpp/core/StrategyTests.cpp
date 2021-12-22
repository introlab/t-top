#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/utils/HbbaLiteException.h>

#include "FilterPoolMock.h"

#include <gtest/gtest.h>

using namespace std;


class StrategyTestee : public Strategy<int>
{
public:
    int onEnablingCount;
    int onDisablingCount;

    StrategyTestee(shared_ptr<FilterPoolMock> filterPool) :
            Strategy(1,
                {{"a", 1}, {"b", 2}},
                {{"c", FilterConfiguration(1)}, {"d", FilterConfiguration(2)}},
                filterPool),
            onEnablingCount(0),
            onDisablingCount(0)
    {
    }
    ~StrategyTestee() override = default;

protected:
    void onEnabling() override
    {
        Strategy::onEnabling();
        onEnablingCount++;
    }

    void onDisabling() override
    {
        Strategy::onDisabling();
        onDisablingCount++;
    }
};

TEST(FilterConfigurationTests, constructor_invalidRate_shouldThrowHbbaLiteException)
{
    EXPECT_THROW(FilterConfiguration(0), HbbaLiteException);
}

TEST(FilterConfigurationTests, getters_shouldReturnTheRightValues)
{
    FilterConfiguration a;
    FilterConfiguration b(10);

    EXPECT_EQ(a.type(), FilterType::ON_OFF);
    EXPECT_EQ(b.type(), FilterType::THROTTLING);
    EXPECT_EQ(b.rate(), 10);
}

TEST(FilterConfigurationTests, equalOperator_shouldReturnTheRightValue)
{
    EXPECT_TRUE(FilterConfiguration() == FilterConfiguration());
    EXPECT_FALSE(FilterConfiguration() == FilterConfiguration(10));
    EXPECT_FALSE(FilterConfiguration(10) == FilterConfiguration());
    EXPECT_FALSE(FilterConfiguration(10) == FilterConfiguration(5));
    EXPECT_TRUE(FilterConfiguration(5) == FilterConfiguration(5));
}

TEST(FilterConfigurationTests, notEqualOperator_shouldReturnTheRightValue)
{
    EXPECT_FALSE(FilterConfiguration() != FilterConfiguration());
    EXPECT_TRUE(FilterConfiguration() != FilterConfiguration(10));
    EXPECT_TRUE(FilterConfiguration(10) != FilterConfiguration());
    EXPECT_TRUE(FilterConfiguration(10) != FilterConfiguration(5));
    EXPECT_FALSE(FilterConfiguration(5) != FilterConfiguration(5));
}

TEST(FilterPoolTests, add_invalidType_shouldThrowHbbaLiteException)
{
    FilterPoolMock testee;
    testee.add("a", FilterType::ON_OFF);
    EXPECT_THROW(testee.add("a", FilterType::THROTTLING), HbbaLiteException);
}

TEST(FilterPoolTests, enable_invalidName_shouldThrowHbbaLiteException)
{
    FilterPoolMock testee;
    EXPECT_THROW(testee.enable("a", FilterConfiguration()), HbbaLiteException);
}

TEST(FilterPoolTests, enable_invalidConfiguration_shouldThrowHbbaLiteException)
{
    FilterPoolMock testee;
    testee.add("a", FilterType::THROTTLING);
    EXPECT_THROW(testee.enable("a", FilterConfiguration()), HbbaLiteException);

    testee.enable("a", FilterConfiguration(10));
    EXPECT_THROW(testee.enable("a", FilterConfiguration(5)), HbbaLiteException);

    testee.disable("a");
    testee.enable("a", FilterConfiguration(5));
    EXPECT_THROW(testee.enable("a", FilterConfiguration(10)), HbbaLiteException);
}

TEST(FilterPoolTests, disable_invalidName_shouldThrowHbbaLiteException)
{
    FilterPoolMock testee;
    EXPECT_THROW(testee.disable("a"), HbbaLiteException);
}

TEST(FilterPoolTests, enableDisable_shouldCallOnMethodOnce)
{
    FilterPoolMock testee;
    testee.add("a", FilterType::THROTTLING);

    testee.enable("a", FilterConfiguration(1));
    EXPECT_EQ(testee.enabledFilters["a"], FilterConfiguration(1));
    EXPECT_EQ(testee.counts["a"], 1);

    testee.enable("a", FilterConfiguration(1));
    EXPECT_EQ(testee.enabledFilters["a"], FilterConfiguration(1));
    EXPECT_EQ(testee.counts["a"], 1);

    testee.disable("a");
    EXPECT_EQ(testee.enabledFilters["a"], FilterConfiguration(1));
    EXPECT_EQ(testee.counts["a"], 1);

    testee.disable("a");
    EXPECT_NE(testee.enabledFilters["a"], FilterConfiguration(1));
    EXPECT_EQ(testee.counts["a"], 0);
}

TEST(StrategyTests, getters_shouldReturnTheRightValues)
{
    const unordered_map<string, uint16_t> EXPECTED_RESOURCES({{"a", 1}, {"b", 2}});
    const unordered_map<string, FilterConfiguration> EXPECTED_FILTER_CONFIGURATIONS({{"c", FilterConfiguration(1)}, {"d", FilterConfiguration(2)}});

    auto filterPool = make_shared<FilterPoolMock>();
    StrategyTestee testee(filterPool);

    EXPECT_EQ(testee.resourcesByName(), EXPECTED_RESOURCES);
    EXPECT_EQ(testee.filterConfigurationsByName(), EXPECTED_FILTER_CONFIGURATIONS);
    EXPECT_EQ(testee.desireType(), type_index(typeid(int)));
}

TEST(StrategyTests, enableDisable_shouldChangeOnceTheState)
{
    auto filterPool = make_shared<FilterPoolMock>();
    StrategyTestee testee(filterPool);
    EXPECT_EQ(testee.onEnablingCount, 0);
    EXPECT_EQ(testee.onDisablingCount, 0);
    EXPECT_FALSE(testee.enabled());

    testee.disable();
    EXPECT_EQ(testee.onEnablingCount, 0);
    EXPECT_EQ(testee.onDisablingCount, 0);
    EXPECT_FALSE(testee.enabled());
    EXPECT_EQ(filterPool->enabledFilters.size(), 0);

    testee.enable();
    EXPECT_EQ(testee.onEnablingCount, 1);
    EXPECT_EQ(testee.onDisablingCount, 0);
    EXPECT_TRUE(testee.enabled());
    EXPECT_EQ(filterPool->enabledFilters["c"], FilterConfiguration(1));
    EXPECT_EQ(filterPool->enabledFilters["d"], FilterConfiguration(2));

    testee.enable();
    EXPECT_EQ(testee.onEnablingCount, 1);
    EXPECT_EQ(testee.onDisablingCount, 0);
    EXPECT_TRUE(testee.enabled());

    testee.disable();
    EXPECT_EQ(testee.onEnablingCount, 1);
    EXPECT_EQ(testee.onDisablingCount, 1);
    EXPECT_FALSE(testee.enabled());
    EXPECT_EQ(filterPool->enabledFilters.size(), 0);

    testee.disable();
    EXPECT_EQ(testee.onEnablingCount, 1);
    EXPECT_EQ(testee.onDisablingCount, 1);
    EXPECT_FALSE(testee.enabled());
}
