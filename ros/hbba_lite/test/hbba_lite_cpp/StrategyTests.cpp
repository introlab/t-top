#include <hbba_lite/Strategy.h>

#include <gtest/gtest.h>

using namespace std;

class StrategyTestee : public Strategy
{
public:
    int onEnableChangedCount;

    StrategyTestee() :
            Strategy(1,
                {{"a", 1}, {"b", 2}},
                {{"c", FilterConfiguration()}, {"d", FilterConfiguration(2)}}),
            onEnableChangedCount(0)
    {
    }
    ~StrategyTestee() override = default;

    type_index desireType() override
    {
        return type_index(typeid(int));
    }

protected:
    void onEnableChanged() override
    {
        onEnableChangedCount++;
    }
};

TEST(FilterConfigurationTests, getters_shouldReturnTheRightValues)
{
    FilterConfiguration a;
    FilterConfiguration b(10);

    EXPECT_FALSE(a.hasRate());
    EXPECT_TRUE(b.hasRate());
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

TEST(StrategyTests, getters_shouldReturnTheRightValues)
{
    const unordered_map<string, uint16_t> EXPECTED_RESSOURCES({{"a", 1}, {"b", 2}});
    const unordered_map<string, FilterConfiguration> EXPECTED_FILTER_CONFIGURATIONS({{"c", FilterConfiguration()}, {"d", FilterConfiguration(2)}});
    StrategyTestee testee;

    EXPECT_EQ(testee.ressourcesByName(), EXPECTED_RESSOURCES);
    EXPECT_EQ(testee.filterConfigurationByName(), EXPECTED_FILTER_CONFIGURATIONS);
}

TEST(StrategyTests, enableDisable_shouldChangeOnceTheState)
{
    StrategyTestee testee;
    EXPECT_EQ(testee.onEnableChangedCount, 0);
    EXPECT_FALSE(testee.enabled());

    testee.disable();
    EXPECT_EQ(testee.onEnableChangedCount, 0);
    EXPECT_FALSE(testee.enabled());

    testee.enable();
    EXPECT_EQ(testee.onEnableChangedCount, 1);
    EXPECT_TRUE(testee.enabled());

    testee.enable();
    EXPECT_EQ(testee.onEnableChangedCount, 1);
    EXPECT_TRUE(testee.enabled());

    testee.disable();
    EXPECT_EQ(testee.onEnableChangedCount, 2);
    EXPECT_FALSE(testee.enabled());

    testee.disable();
    EXPECT_EQ(testee.onEnableChangedCount, 2);
    EXPECT_FALSE(testee.enabled());
}
