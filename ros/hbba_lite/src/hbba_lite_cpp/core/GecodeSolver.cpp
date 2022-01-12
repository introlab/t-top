#include <hbba_lite/core/GecodeSolver.h>

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

using namespace std;
using namespace Gecode;

constexpr int MAX_INT_VAR_VALUE = 1000000;

class GecodeSolverSpace : public IntMaximizeSpace {
    const vector<unique_ptr<Desire>>& m_desires;
    const vector<size_t>& m_mostIntenseDesiresIndexes;
    const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& m_strategiesByDesireType;
    const unordered_map<string, uint16_t>& m_systemResourcesByName;

    // Branching variables
    BoolVarArray m_isDesireEnabled;
    IntVarArray m_strategyIndexes;

    // Intermediate variables
    IntVar m_zero;
    IntVar m_minusOne;
    unordered_map<string, IntVar> m_ressourcesValuesByName;
    unordered_map<string, IntVar> m_filterRatesByName;

    // Cost variable
    IntVar m_costValue;

public:
    GecodeSolverSpace(const vector<unique_ptr<Desire>>& desires,
        const vector<size_t>& mostIntenseDesiresIndexes,
        const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType,
        const unordered_map<string, uint16_t>& systemResourcesByName);
    GecodeSolverSpace(GecodeSolverSpace& other);

    Space* copy() override;
    IntVar cost() const override;

    unordered_set<SolverResult> convert();

private:
    void setupStrategyIndexDomains();
    void setupResourceContraints();
    void setupFilterRateContraints();
    void setupCostValue();

    IntArgs getUtilityIntArgs(const vector<unique_ptr<BaseStrategy>>& strategies);
    IntArgs getResourcesIntArgs(const string& name, const vector<unique_ptr<BaseStrategy>>& strategies);

    vector<string> getAllFilterNames();
    IntArgs getFilterRateIntArgs(const string& name, const vector<unique_ptr<BaseStrategy>>& strategies);
};

GecodeSolverSpace::GecodeSolverSpace(const vector<unique_ptr<Desire>>& desires,
    const vector<size_t>& mostIntenseDesiresIndexes,
    const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType,
    const unordered_map<string, uint16_t>& systemResourcesByName) :
        m_desires(desires),
        m_mostIntenseDesiresIndexes(mostIntenseDesiresIndexes),
        m_strategiesByDesireType(strategiesByDesireType),
        m_systemResourcesByName(systemResourcesByName),
        m_isDesireEnabled(*this, mostIntenseDesiresIndexes.size(), 0, 1),
        m_strategyIndexes(*this, mostIntenseDesiresIndexes.size(), 0, MAX_INT_VAR_VALUE),
        m_zero(*this, IntSet({0})),
        m_minusOne(*this, IntSet({-1})),
        m_costValue(*this, 0, MAX_INT_VAR_VALUE)
{
    setupStrategyIndexDomains();
    setupResourceContraints();
    setupFilterRateContraints();
    setupCostValue();

    branch(*this, m_isDesireEnabled, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    branch(*this, m_strategyIndexes, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
}

GecodeSolverSpace::GecodeSolverSpace(GecodeSolverSpace& other) : IntMaximizeSpace(other),
        m_desires(other.m_desires),
        m_mostIntenseDesiresIndexes(other.m_mostIntenseDesiresIndexes),
        m_strategiesByDesireType(other.m_strategiesByDesireType),
        m_systemResourcesByName(other.m_systemResourcesByName)
{
    m_isDesireEnabled.update(*this, other.m_isDesireEnabled);
    m_strategyIndexes.update(*this, other.m_strategyIndexes);

    m_zero.update(*this, other.m_zero);
    m_minusOne.update(*this, other.m_minusOne);

    for (auto& pair : other.m_ressourcesValuesByName)
    {
        m_ressourcesValuesByName[pair.first].update(*this, pair.second);
    }
    for (auto& pair : other.m_filterRatesByName)
    {
        m_filterRatesByName[pair.first].update(*this, pair.second);
    }

    m_costValue.update(*this, other.m_costValue);
}

Space* GecodeSolverSpace::copy()
{
    return new GecodeSolverSpace(*this);
}

IntVar GecodeSolverSpace::cost() const
{
    return m_costValue;
}

unordered_set<SolverResult> GecodeSolverSpace::convert()
{
    unordered_set<SolverResult> results;
    for (size_t i = 0; i < m_mostIntenseDesiresIndexes.size(); i++)
    {
        if (m_isDesireEnabled[i].val())
        {
            results.emplace(m_mostIntenseDesiresIndexes[i], m_strategyIndexes[i].val());
        }
    }

    return results;
}

void GecodeSolverSpace::setupStrategyIndexDomains()
{
    for (size_t i = 0; i < m_mostIntenseDesiresIndexes.size(); i++)
    {
        auto desireType = m_desires[m_mostIntenseDesiresIndexes[i]]->type();
        size_t strategyCount = m_strategiesByDesireType.at(desireType).size();
        dom(*this, m_strategyIndexes[i], 0, strategyCount - 1);
    }
}

void GecodeSolverSpace::setupResourceContraints()
{
    unordered_map<string, IntVarArgs> resourcesValuesByName;
    for (size_t i = 0; i < m_mostIntenseDesiresIndexes.size(); i++)
    {
        auto desireType = m_desires[m_mostIntenseDesiresIndexes[i]]->type();

        for (auto& pair : m_systemResourcesByName)
        {
            IntArgs resources = getResourcesIntArgs(pair.first, m_strategiesByDesireType.at(desireType));
            IntVar resource(*this, 0, MAX_INT_VAR_VALUE);
            element(*this, resources, m_strategyIndexes[i], resource);

            IntVar resourceWithState(*this, 0, MAX_INT_VAR_VALUE);
            ite(*this, m_isDesireEnabled[i], resource, m_zero, resourceWithState);

            resourcesValuesByName[pair.first] << resourceWithState;
        }
    }

    for (auto& pair : resourcesValuesByName)
    {
        m_ressourcesValuesByName[pair.first] = IntVar(*this, 0, MAX_INT_VAR_VALUE);
        linear(*this, pair.second, IRT_EQ, m_ressourcesValuesByName[pair.first]);
        rel(*this, m_ressourcesValuesByName[pair.first] <= m_systemResourcesByName.at(pair.first));
    }
}

void GecodeSolverSpace::setupFilterRateContraints()
{
    auto allFilterNames = getAllFilterNames();
    unordered_map<string, IntVarArgs> filterRatesByName;

    for (size_t i = 0; i < m_mostIntenseDesiresIndexes.size(); i++)
    {
        auto desireType = m_desires[m_mostIntenseDesiresIndexes[i]]->type();

        for (auto& name : allFilterNames)
        {
            IntArgs filterRates = getFilterRateIntArgs(name, m_strategiesByDesireType.at(desireType));
            IntVar filterRate(*this, -1, MAX_INT_VAR_VALUE);
            element(*this, filterRates, m_strategyIndexes[i], filterRate);

            IntVar filterRateWithState(*this, -1, MAX_INT_VAR_VALUE);
            ite(*this, m_isDesireEnabled[i], filterRate, m_minusOne, filterRateWithState);

            filterRatesByName[name] << filterRateWithState;
        }
    }

    for (auto& pair : filterRatesByName)
    {
        m_filterRatesByName[pair.first] = IntVar(*this, -1, MAX_INT_VAR_VALUE);
        max(*this, pair.second, m_filterRatesByName[pair.first]);

        IntVar minusOneCount(*this, 0, MAX_INT_VAR_VALUE);
        IntVar maxCount(*this, 0, MAX_INT_VAR_VALUE);
        count(*this, pair.second, -1, IRT_EQ, minusOneCount);
        count(*this, pair.second, m_filterRatesByName[pair.first], IRT_EQ, maxCount);

        rel(*this, filterRatesByName[pair.first].size() == minusOneCount + maxCount ||
            2 * filterRatesByName[pair.first].size() == minusOneCount + maxCount);
    }
}

void GecodeSolverSpace::setupCostValue()
{
    IntVarArray utilityValues(*this, m_mostIntenseDesiresIndexes.size(), 0, MAX_INT_VAR_VALUE);
    IntVarArray utilityIntensityProduct(*this, m_mostIntenseDesiresIndexes.size(), 0, MAX_INT_VAR_VALUE);
    IntVarArray costValues(*this, m_mostIntenseDesiresIndexes.size(), 0, MAX_INT_VAR_VALUE);

    for (size_t i = 0; i < m_mostIntenseDesiresIndexes.size(); i++)
    {
        auto desireType = m_desires[m_mostIntenseDesiresIndexes[i]]->type();
        int desireIntensity = m_desires[m_mostIntenseDesiresIndexes[i]]->intensity();

        IntArgs utilities = getUtilityIntArgs(m_strategiesByDesireType.at(desireType));
        element(*this, utilities, m_strategyIndexes[i], utilityValues[i]);
        rel(*this, utilityIntensityProduct[i] == utilityValues[i] * desireIntensity);
        ite(*this, m_isDesireEnabled[i], utilityIntensityProduct[i], m_zero, costValues[i]);
    }
    linear(*this, costValues, IRT_EQ, m_costValue);
}

IntArgs GecodeSolverSpace::getUtilityIntArgs(const vector<unique_ptr<BaseStrategy>>& strategies)
{
    IntArgs utilities(strategies.size());
    for (size_t i = 0; i < strategies.size(); i++)
    {
        utilities[i] = strategies[i]->utility();
    }
    return utilities;
}

IntArgs GecodeSolverSpace::getResourcesIntArgs(const string& name, const vector<unique_ptr<BaseStrategy>>& strategies)
{
    IntArgs resources(strategies.size());
    for (size_t i = 0; i < strategies.size(); i++)
    {
        auto& resourcesByName = strategies[i]->resourcesByName();
        auto it = resourcesByName.find(name);
        if (it == resourcesByName.end())
        {
            resources[i] = 0;
        }
        else
        {
            resources[i] = it->second;
        }

    }
    return resources;
}

vector<string> GecodeSolverSpace::getAllFilterNames()
{
    unordered_set<string> allFilterNames;

    for (auto& pair : m_strategiesByDesireType)
    {
        for (auto& strategy : pair.second)
        {
            for (auto filter : strategy->filterConfigurationsByName())
            {
                allFilterNames.emplace(filter.first);
            }
        }
    }

    return vector<string>(allFilterNames.begin(), allFilterNames.end());
}

IntArgs GecodeSolverSpace::getFilterRateIntArgs(const string& name, const vector<unique_ptr<BaseStrategy>>& strategies)
{
    IntArgs filterRates(strategies.size());
    for (size_t i = 0; i < strategies.size(); i++)
    {
        auto& filterConfigurationsByName = strategies[i]->filterConfigurationsByName();
        auto it = filterConfigurationsByName.find(name);
        if (it == filterConfigurationsByName.end())
        {
            filterRates[i] = -1;
        }
        else
        {
            filterRates[i] = it->second.rate();
        }

    }

    return filterRates;
}


GecodeSolver::GecodeSolver()
{
}

unordered_set<SolverResult> GecodeSolver::solve(const vector<unique_ptr<Desire>>& desires,
    const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType,
    const unordered_map<string, uint16_t>& systemResourcesByName)
{
    if (desires.empty())
    {
        return {};
    }

    checkDesireStrategies(desires, strategiesByDesireType);
    checkStrategyResources(strategiesByDesireType, systemResourcesByName);

    auto mostIntenseDesiresIndexes = selectMostIntenseEnabledDesireIndexes(desires);

    GecodeSolverSpace model(desires, mostIntenseDesiresIndexes, strategiesByDesireType, systemResourcesByName);
    BAB<GecodeSolverSpace> e(&model);

    unique_ptr<GecodeSolverSpace> bestSpace;
    while (GecodeSolverSpace* space = e.next())
    {
        bestSpace = unique_ptr<GecodeSolverSpace>(space);
    }

    if (bestSpace)
    {
        return bestSpace->convert();
    }
    else
    {
        return {};
    }
}
