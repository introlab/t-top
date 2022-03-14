#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/utils/HbbaLiteException.h>
#include <hbba_lite/core/GecodeSolver.h>

#include "Desires.h"
#include "FilterPoolMock.h"

#include <gtest/gtest.h>

using namespace std;


bool contains(const std::string& data, const std::string& substr)
{
    return data.find(substr) != std::string::npos;
}

class SolverMock : public Solver
{
public:
    SolverMock()
    {
    }
    ~SolverMock() override = default;

    // Return the strategy to activate (Desire Type, strategy index)
    virtual unordered_set<SolverResult> solve(const vector<unique_ptr<Desire>>& desires,
        const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType,
        const unordered_map<string, uint16_t>& systemResourcesByName)
    {
        if (desires.empty())
        {
            return {};
        }

        EXPECT_EQ(desires.size(), 1);
        if (desires.size() == 1)
        {
            EXPECT_EQ(desires[0]->type(), type_index(typeid(DesireD)));
        }

        EXPECT_EQ(strategiesByDesireType.size(), 1);
        auto it = strategiesByDesireType.find(type_index(typeid(DesireD)));
        if (it != strategiesByDesireType.end())
        {
            EXPECT_EQ(it->second.size(), 1);
            if (it->second.size() == 1)
            {
                EXPECT_EQ(it->second[0]->desireType(), type_index(typeid(DesireD)));
            }
        }
        else
        {
            ADD_FAILURE();
        }

        const unordered_map<string, uint16_t> EXPECTED_RESSOURCES({{"ra", 10}});
        EXPECT_EQ(systemResourcesByName, EXPECTED_RESSOURCES);

        return {SolverResult(0, 0)};
    }
};

TEST(HbbaLiteTests, constructor_invalidResourceName_shouldThrowHbbaLiteException)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategy = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto solver = make_unique<SolverMock>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategy));


    EXPECT_THROW(HbbaLite(desireSet, move(strategies), {}, move(solver)), HbbaLiteException);
}

TEST(HbbaLiteTests, constructor_invalidResourceCount_shouldThrowHbbaLiteException)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategy = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto solver = make_unique<SolverMock>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategy));


    EXPECT_THROW(HbbaLite(desireSet, move(strategies), {{"ra", 9}}, move(solver)), HbbaLiteException);
}

TEST(HbbaLiteTests, onDesireSetChange_shouldEnableDisableStrategies)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategy = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto solver = make_unique<SolverMock>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategy));

    HbbaLite testee(desireSet, move(strategies), {{"ra", 10}}, move(solver));

    auto desire = make_unique<DesireD>();
    auto id = desire->id();
    desireSet->addDesire(move(desire));
    this_thread::sleep_for(10ms);
    EXPECT_EQ(filterPool->counts["fa"], 1);

    desireSet->removeDesire(id);
    this_thread::sleep_for(10ms);
    EXPECT_EQ(filterPool->counts["fa"], 0);
}

TEST(HbbaLiteTests, getActiveStrategies_shouldReturnActiveStrategies)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategyC = make_unique<Strategy<DesireC>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyD = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{
                {"fb", FilterConfiguration::throttling(1)},
                {"fc", FilterConfiguration::onOff()}
            },
        filterPool);
    auto solver = make_unique<GecodeSolver>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategyC));
    strategies.emplace_back(move(strategyD));

    HbbaLite testee(desireSet, move(strategies), {{"ra", 10}}, move(solver));

    auto desire = make_unique<DesireC>();
    auto id = desire->id();
    desireSet->addDesire(move(desire));
    this_thread::sleep_for(10ms);

    std::set<std::string> expectedStrategies = {
        std::string(type_index(typeid(DesireC)).name()).append("::(u:10; r:{ra:10}; f:{fa:THROTTLING=1})"),
    };
    EXPECT_EQ(testee.getActiveStrategies(), expectedStrategies);

    {
        desireSet->beginTransaction();
        desireSet->removeDesire(id);

        auto desire2 = make_unique<DesireD>();
        auto id2 = desire2->id();
        desireSet->addDesire(move(desire2));
    }

    this_thread::sleep_for(10ms);
    std::set<std::string> expectedStrategies2 = {
        std::string(type_index(typeid(DesireD)).name()).append("::(u:10; r:{ra:10}; f:{fb:THROTTLING=1; fc:ON_OFF})"),
    };
    auto actualStrategies = testee.getActiveStrategies();
    ASSERT_EQ(actualStrategies.size(), 1);
    EXPECT_TRUE(contains(*actualStrategies.begin(), "::(u:10; r:{ra:10}; f:{"));
    EXPECT_TRUE(contains(*actualStrategies.begin(), "fb:THROTTLING=1"));
    EXPECT_TRUE(contains(*actualStrategies.begin(), "fc:ON_OFF"));
}

TEST(HbbaLiteTests, getActiveDesireNames_shouldReturnActiveDesireName)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategyC = make_unique<Strategy<DesireC>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyD = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 10}},
        unordered_map<string, FilterConfiguration>{
                {"fb", FilterConfiguration::throttling(1)},
                {"fc", FilterConfiguration::onOff()}
            },
        filterPool);
    auto solver = make_unique<GecodeSolver>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategyC));
    strategies.emplace_back(move(strategyD));

    HbbaLite testee(desireSet, move(strategies), {{"ra", 10}}, move(solver));

    auto desire = make_unique<DesireC>();
    auto id = desire->id();
    desireSet->addDesire(move(desire));
    this_thread::sleep_for(10ms);

    std::set<std::string> expectedDesireNames = {
        std::string(type_index(typeid(DesireC)).name()),
    };
    EXPECT_EQ(testee.getActiveDesireNames(), expectedDesireNames);

    {
        desireSet->beginTransaction();
        desireSet->removeDesire(id);

        auto desire2 = make_unique<DesireD>();
        auto id2 = desire2->id();
        desireSet->addDesire(move(desire2));
    }

    this_thread::sleep_for(10ms);
    std::set<std::string> expectedDesireNames2 = {
        std::string(type_index(typeid(DesireD)).name()),
    };
    EXPECT_EQ(testee.getActiveDesireNames(), expectedDesireNames2);
}

TEST(HbbaLiteTests, getActiveDesireNames_shouldOnlyReturnDesireNameWithBiggestIntensity)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<FilterPoolMock>();
    auto strategyB = make_unique<Strategy<DesireB>>(10,
        unordered_map<string, uint16_t>{{"ra", 1}},
        unordered_map<string, FilterConfiguration>{{"fa", FilterConfiguration::throttling(1)}},
        filterPool);
    auto strategyC = make_unique<Strategy<DesireC>>(10,
        unordered_map<string, uint16_t>{{"ra", 1}},
        unordered_map<string, FilterConfiguration>{{"fd", FilterConfiguration::onOff()}},
        filterPool);
    auto strategyD = make_unique<Strategy<DesireD>>(10,
        unordered_map<string, uint16_t>{{"ra", 1}},
        unordered_map<string, FilterConfiguration>{
                {"fb", FilterConfiguration::throttling(1)},
                {"fc", FilterConfiguration::onOff()}
            },
        filterPool);
    auto solver = make_unique<GecodeSolver>();

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(move(strategyB));
    strategies.emplace_back(move(strategyC));
    strategies.emplace_back(move(strategyD));

    HbbaLite testee(desireSet, move(strategies), {{"ra", 1}}, move(solver));

    {
        desireSet->beginTransaction();

        auto desireB = make_unique<DesireB>(2);
        auto idB = desireB->id();
        desireSet->addDesire(move(desireB));

        auto desireC = make_unique<DesireC>();
        auto idC = desireC->id();
        desireSet->addDesire(move(desireC));

        auto desireD = make_unique<DesireD>();
        auto idD = desireD->id();
        desireSet->addDesire(move(desireD));
    }

    this_thread::sleep_for(10ms);
    std::set<std::string> expectedDesireNames = {
        std::string(type_index(typeid(DesireB)).name()),
    };
    EXPECT_EQ(testee.getActiveDesireNames(), expectedDesireNames);
}
