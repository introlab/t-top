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
    unordered_set<pair<type_index, size_t>> enabledStrategies;
    vector<tuple<type_index, size_t, unique_ptr<Desire>&>> strategiesToEnable;

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

        if (toBeEnabled || strategy->enabled() && strategy->desireId() != desire->id())
        {
            strategiesToEnable.emplace_back(desireType, result.strategyIndex, desire);
        }
        else
        {
            enabledStrategies.erase(p);
        }
    }

    for (auto& p : enabledStrategies)
    {
        m_strategiesByDesireType[p.first][p.second]->disable();
    }
    for (auto& s : strategiesToEnable)
    {
        m_strategiesByDesireType[get<0>(s)][get<1>(s)]->enable(get<2>(s));
    }

    std::lock_guard<std::mutex> lock(m_activeDesireNamesMutex);
    m_activeDesireNames.clear();
    for (auto& s : strategiesToEnable)
    {
        m_activeDesireNames.emplace(get<0>(s).name());
    }
}

std::vector<std::string> HbbaLite::getActiveStrategies() const
{
    vector<string> activeStrategies;

    for (const auto& p : m_strategiesByDesireType)
    {
        for (const auto& i : p.second)
        {
            if (i->enabled())
            {
                for (const auto& p2 : i->filterConfigurationsByName())
                {
                    activeStrategies.emplace_back(string(p.first.name()) + "::" + p2.first
                                                  + "=" + to_string(i->enabled()));
                }
                break;
            }
        }
    }

    return activeStrategies;
}

std::vector<std::string> HbbaLite::getActiveDesireNames() const
{
    std::lock_guard<std::mutex> lock(m_activeDesireNamesMutex);
    return {std::begin(m_activeDesireNames), std::end(m_activeDesireNames)};
}
