#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/utils/HbbaLiteException.h>

using namespace std;

HbbaLite::HbbaLite(shared_ptr<DesireSet> desireSet,
                   vector<unique_ptr<BaseStrategy>> strategies,
                   unordered_map<string, uint16_t> resourcesByNames,
                   unique_ptr<Solver> solver)
    : m_desireSet(move(desireSet)), m_resourcesByNames(move(resourcesByNames)),
      m_solver(move(solver)), m_pendingDesiresSemaphore(false), m_stopped(false)
{
    for (auto& strategy : strategies)
    {
        checkStrategyResources(strategy->desireType(), strategy->resourcesByName());
        m_strategiesByDesireType[strategy->desireType()].emplace_back(move(strategy));
    }

    m_thread = make_unique<thread>(&HbbaLite::run, this);
    m_desireSet->addObserver(this);
}

HbbaLite::~HbbaLite()
{
    m_desireSet->removeObserver(this);

    m_stopped.store(true);
    m_thread->join();
}

void HbbaLite::onDesireSetChanged(const vector<unique_ptr<Desire>>& enabledDesires)
{
    lock_guard<mutex> lock(m_pendingDesiresMutex);
    m_pendingDesires.clear();

    for (auto& enabledDesire : enabledDesires)
    {
        m_pendingDesires.emplace_back(enabledDesire->clone());
    }

    m_pendingDesiresSemaphore.release();
}

void HbbaLite::checkStrategyResources(
    type_index desireType, const unordered_map<string, uint16_t>& resourcesByNames)
{
    for (auto& pair : resourcesByNames)
    {
        auto it = m_resourcesByNames.find(pair.first);
        if (it == m_resourcesByNames.end())
        {
            throw HbbaLiteException("A strategy for \"" + string(desireType.name())
                                    + "\" desires have an invalid resource (" + pair.first
                                    + ").");
        }
        else if (pair.second > it->second)
        {
            throw HbbaLiteException("A strategy for \"" + string(desireType.name())
                                    + "\" desires have an invalid resource count ("
                                    + pair.first + "=" + to_string(pair.second) + ").");
        }
    }
}

void HbbaLite::run()
{
    constexpr chrono::milliseconds SEMAPHORE_WAIT_DURATION(10);

    while (!m_stopped.load())
    {
        if (m_pendingDesiresSemaphore.tryAcquireFor(SEMAPHORE_WAIT_DURATION))
        {
            vector<unique_ptr<Desire>> desires;
            {
                lock_guard<mutex> lock(m_pendingDesiresMutex);
                swap(m_pendingDesires, desires);
            }
            updateStrategies(move(desires));
        }
    }
}

void HbbaLite::updateStrategies(vector<unique_ptr<Desire>> desires)
{
    auto results = m_solver->solve(desires, m_strategiesByDesireType, m_resourcesByNames);
    
    // After the loop iterating the results, this set will contain the strategies that need to be disabled.
    unordered_set<pair<type_index, size_t>> enabledStrategies;
    
    // After the loop iterating the results, this vector will contain the strategies that need to be enabled.
    vector<pair<size_t, unique_ptr<Desire>&>> strategiesToEnable;

    for (auto& p : m_strategiesByDesireType)
    {
        for (size_t i = 0; i < p.second.size(); i++)
        {
            if (p.second[i]->enabled())
            {
                enabledStrategies.emplace(p.first, i);
            }
        }
    }

    for (auto result : results)
    {
        auto& desire = desires[result.desireIndex];
        auto desireType = desire->type();
        auto p = pair<type_index, size_t>(desireType, result.strategyIndex);
        bool toBeEnabled = enabledStrategies.count(p) == 0;
        auto& strategy = m_strategiesByDesireType[p.first][p.second];

        if (
                // The strategy must be enabled, but it is disabled
                toBeEnabled ||
                // The strategy is already enabled for another desire, so it must be disabled then enabled.
                strategy->enabled() && strategy->desireId() != desire->id()
            )
        {
            strategiesToEnable.emplace_back(result.strategyIndex, desire);
        }
        else
        {
            enabledStrategies.erase(p);
        }
    }

    for (const auto& p : enabledStrategies)
    {
        m_strategiesByDesireType[p.first][p.second]->disable();
    }
    for (const auto& s : strategiesToEnable)
    {
        m_strategiesByDesireType[s.second->type()][s.first]->enable(s.second);
    }

    updateActiveDesireNames(desires, results);
    updateActiveStrategies(desires, results);
}

void HbbaLite::updateActiveDesireNames(
    const std::vector<std::unique_ptr<Desire>>& desires,
    const std::unordered_set<SolverResult>& results)
{
    std::lock_guard<std::mutex> lock(m_activeDesireNamesMutex);
    m_activeDesireNames.clear();
    for (auto result : results)
    {
        m_activeDesireNames.emplace(desires[result.desireIndex]->type().name());
    }
}

void HbbaLite::updateActiveStrategies(const std::vector<std::unique_ptr<Desire>>& desires,
                                      const std::unordered_set<SolverResult>& results)
{
    std::lock_guard<std::mutex> lock(m_activeStrategiesMutex);
    m_activeStrategies.clear();
    for (auto result : results)
    {
        const auto& desire = desires[result.desireIndex];
        const auto& strategy =
            m_strategiesByDesireType[desire->type()][result.strategyIndex];
        std::string desireName = desire->type().name();
        std::string strategyUtility = std::to_string(strategy->utility());
        std::string s = desireName.append("::(u:")
                            .append(strategyUtility)
                            .append("; ")
                            .append("r:{");

        size_t counter = strategy->resourcesByName().size();
        for (const auto& resource : strategy->resourcesByName())
        {
            s.append(resource.first).append(":").append(std::to_string(resource.second));
            if (--counter != 0)
            {
                s.append("; ");
            }
        }

        s.append("}; f:{");

        counter = strategy->filterConfigurationsByName().size();
        for (const auto& filter : strategy->filterConfigurationsByName())
        {
            s.append(filter.first).append(":").append(filterTypeToString(filter.second.type()));
            if (filter.second.type() == FilterType::THROTTLING)
            {
                s.append("=").append(std::to_string(filter.second.rate()));
            }                
            if (--counter != 0)
            {
                s.append("; ");
            }
        }
        s.append("})");
        m_activeStrategies.emplace(std::move(s));
    }
}

std::set<std::string> HbbaLite::getActiveStrategies() const
{
    std::lock_guard<std::mutex> lock(m_activeStrategiesMutex);
    return m_activeStrategies;
    // set<string> activeStrategies;

    // for (const auto& p : m_strategiesByDesireType)
    // {
    //     for (const auto& i : p.second)
    //     {
    //         if (i->enabled())
    //         {
    //             for (const auto& p2 : i->filterConfigurationsByName())
    //             {
    //                 activeStrategies.emplace(string(p.first.name()) + "::" + p2.first
    //                                          + "=" + to_string(i->enabled()));
    //             }
    //             break;
    //         }
    //     }
    // }

    // return activeStrategies;
}

std::set<std::string> HbbaLite::getActiveDesireNames() const
{
    std::lock_guard<std::mutex> lock(m_activeDesireNamesMutex);
    return m_activeDesireNames;
}
