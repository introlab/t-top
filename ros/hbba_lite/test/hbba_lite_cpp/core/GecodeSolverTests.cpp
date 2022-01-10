#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/utils/HbbaLiteException.h>

#include "Desires.h"
#include "FilterPoolMock.h"

#include <gtest/gtest.h>

using namespace std;

TEST(GecodeSolverTests, solve_missingStrategy_shouldThrowHbbaLiteException)
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

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}, {"rb", 30}};

    GecodeSolver testee;
    EXPECT_THROW(testee.solve(desires, strategiesByDesireType, systemResourcesByName), HbbaLiteException);
}

TEST(GecodeSolverTests, solve_missingRessource_shouldThrowHbbaLiteException)
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
        unordered_map<string, uint16_t>{{"ra", 10}, {"rb", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}};

    GecodeSolver testee;
    EXPECT_THROW(testee.solve(desires, strategiesByDesireType, systemResourcesByName), HbbaLiteException);
}

TEST(GecodeSolverTests, solve_compatibleFilters1_shouldReturnStrategiesToActivate)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(2));
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(3));
    desires.emplace_back(make_unique<DesireC>());

    auto strategyA1 = make_unique<Strategy<DesireA>>(1,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyA2 = make_unique<Strategy<DesireA>>(2,
        unordered_map<string, uint16_t>{{"ra", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyA3 = make_unique<Strategy<DesireA>>(3,
        unordered_map<string, uint16_t>{{"ra", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB1 = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}, {"rb", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB2 = make_unique<Strategy<DesireB>>(1,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyC = make_unique<Strategy<DesireC>>(1,
        unordered_map<string, uint16_t>{{"ra", 20}, {"rb", 30}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA1));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA2));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA3));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB1));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB2));
    strategiesByDesireType[type_index(typeid(DesireC))].emplace_back(move(strategyC));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}, {"rb", 30}};

    GecodeSolver testee;
    auto result = testee.solve(desires, strategiesByDesireType, systemResourcesByName);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result.count(SolverResult(0, 0)), 1);
    EXPECT_EQ(result.count(SolverResult(2, 0)), 1);
}

TEST(GecodeSolverTests, solve_compatibleFilters2_shouldReturnStrategiesToActivate)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(2));
    desires.emplace_back(make_unique<DesireB>(3));
    desires.emplace_back(make_unique<DesireC>());

    auto strategyA1 = make_unique<Strategy<DesireA>>(1,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyA2 = make_unique<Strategy<DesireA>>(2,
        unordered_map<string, uint16_t>{{"ra", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyA3 = make_unique<Strategy<DesireA>>(3,
        unordered_map<string, uint16_t>{{"ra", 20}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB1 = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}, {"rb", 20}},
        unordered_map<string, FilterConfiguration>{{"fb", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB2 = make_unique<Strategy<DesireB>>(1,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fb", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyC = make_unique<Strategy<DesireC>>(50,
        unordered_map<string, uint16_t>{{"ra", 20}, {"rb", 30}},
        unordered_map<string, FilterConfiguration>{{"fc", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA1));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA2));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA3));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB1));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB2));
    strategiesByDesireType[type_index(typeid(DesireC))].emplace_back(move(strategyC));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}, {"rb", 30}};

    GecodeSolver testee;
    auto result = testee.solve(desires, strategiesByDesireType, systemResourcesByName);

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result.count(SolverResult(2, 0)), 1);
}

TEST(GecodeSolverTests, solve_incompatibleFilters1_shouldReturnStrategiesToActivate)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(1));

    auto strategyA1 = make_unique<Strategy<DesireA>>(1,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(2)}},
        filterPool);
    auto strategyA2 = make_unique<Strategy<DesireA>>(3,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyB1 = make_unique<Strategy<DesireB>>(1,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(2)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA1));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA2));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB1));

    unordered_map<string, uint16_t> systemResourcesByName = {};

    GecodeSolver testee;
    auto result = testee.solve(desires, strategiesByDesireType, systemResourcesByName);

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result.count(SolverResult(0, 1)), 1);
}

TEST(GecodeSolverTests, solve_incompatibleFilters2_shouldReturnStrategiesToActivate)
{
    auto filterPool = make_shared<FilterPoolMock>();

    vector<unique_ptr<Desire>> desires;
    desires.emplace_back(make_unique<DesireA>(1));
    desires.emplace_back(make_unique<DesireB>(1));
    desires.emplace_back(make_unique<DesireC>());

    auto strategyA1 = make_unique<Strategy<DesireA>>(2,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::onOff()}, {"fb", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyA2 = make_unique<Strategy<DesireA>>(5,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::onOff()}},
        filterPool);
    auto strategyB1 = make_unique<Strategy<DesireB>>(3,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::onOff()}, {"fb", FilterConfiguration::throttling(2)}},
        filterPool);
    auto strategyB = make_unique<Strategy<DesireB>>(2,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::onOff()}},
        filterPool);
    auto strategyC = make_unique<Strategy<DesireC>>(50,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fc", FilterConfiguration::throttling(1)}},
        filterPool);

    unordered_map<type_index, vector<unique_ptr<BaseStrategy>>> strategiesByDesireType;
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA1));
    strategiesByDesireType[type_index(typeid(DesireA))].emplace_back(move(strategyA2));
    strategiesByDesireType[type_index(typeid(DesireB))].emplace_back(move(strategyB1));
    strategiesByDesireType[type_index(typeid(DesireC))].emplace_back(move(strategyC));

    unordered_map<string, uint16_t> systemResourcesByName = {{"ra", 20}};

    GecodeSolver testee;
    auto result = testee.solve(desires, strategiesByDesireType, systemResourcesByName);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result.count(SolverResult(0, 1)), 1);
    EXPECT_EQ(result.count(SolverResult(2, 0)), 1);
}
