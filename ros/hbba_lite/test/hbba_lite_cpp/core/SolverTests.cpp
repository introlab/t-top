#include <hbba_lite/core/Solver.h>
#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/utils/HbbaLiteException.h>

#include "Desires.h"
#include "FilterPoolMock.h"

#include <gtest/gtest.h>

using namespace std;

TEST(SolverTests, checkDesireStrategies_missingStrategy_shouldThrowHbbaLiteException)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(1));

    auto strategyA = make_unique<Strategy<DesireA>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));

    EXPECT_THROW(checkDesireStrategies(desires, strategiesByDesireType), HbbaLiteException);
}

TEST(SolverTests, checkDesireStrategies_emptyStrategies_shouldThrowHbbaLiteException)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(1));

    auto strategyA = make_unique<Strategy<DesireA>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));
    strategiesByDesireType[type_index(typeid(DesireB))].clear();

    EXPECT_THROW(checkDesireStrategies(desires, strategiesByDesireType), HbbaLiteException);
}

TEST(SolverTests, checkDesireStrategies_shouldNotThrowHbbaLiteException)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(1));

    auto strategyA = make_unique<Strategy<DesireA>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB));

    checkDesireStrategies(desires, strategiesByDesireType);
}

TEST(SolverTests, checkStrategyResources_missingRessource_shouldThrowHbbaLiteException)
{
    auto filterPool = make_shared<FilterPoolMock>();

    auto strategyA = make_unique<Strategy<DesireA>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}, {"rb", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}};

    EXPECT_THROW(checkStrategyResources(strategiesByDesireType, systemResourcesByName), HbbaLiteException);
}

TEST(SolverTests, checkStrategyResources_shouldNotThrowHbbaLiteException)
{
    auto filterPool = make_shared<FilterPoolMock>();

    auto strategyA = make_unique<Strategy<DesireA>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}, {"rb", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}, {"rb", 30}};

    checkStrategyResources(strategiesByDesireType, systemResourcesByName);
}

TEST(SolverTests, selectMostIntenseEnabledDesireIndexes_shouldReturnMostIntenseDesireIndexes)
{
    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireA>(2));
    desires.emplace_back(make_unique<DesireB>(3));
    desires.emplace_back(make_unique<DesireB>(2));
    desires.emplace_back(make_unique<DesireB>(1));
    desires.emplace_back(make_unique<DesireC>());

    desires.emplace_back(make_unique<DesireB>(4));
    desires[6]->disable();

    auto mostIntenseDesires = selectMostIntenseEnabledDesireIndexes(desires);

    EXPECT_EQ(mostIntenseDesires, vector<size_t>({5, 2, 1}));
}
