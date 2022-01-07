#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/utils/HbbaLiteException.h>

#include "Desires.h"
#include "FilterPoolMock.h"

#include <gtest/gtest.h>

using namespace std;

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
