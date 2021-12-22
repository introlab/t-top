#include <hbba_lite/core/HbbaLite.h>

#include <hbba_lite/utils/HbbaLiteException.h>

using namespace std;

HbbaLite::HbbaLite(shared_ptr<DesireSet> desireSet,
    vector<unique_ptr<BaseStrategy>> strategies,
    unordered_map<string, uint16_t> resourcesByNames,
    unique_ptr<Solver> solver) :
        m_desireSet(move(desireSet)),
        m_resourcesByNames(move(resourcesByNames)),
        m_solver(move(solver)),
        m_pendingDesiresSemaphore(false),
        m_stopped(false)
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

void HbbaLite::checkStrategyResources(type_index desireType, const unordered_map<string, uint16_t>& resourcesByNames)
{
    for (auto& pair : resourcesByNames)
    {
        auto it = m_resourcesByNames.find(pair.first);
        if (it == m_resourcesByNames.end())
        {
            throw HbbaLiteException("A strategy for \"" + string(desireType.name()) +
                "\" desires have an invalid resource (" + pair.first + ").");
        }
        else if (pair.second > it->second)
        {
            throw HbbaLiteException("A strategy for \"" + string(desireType.name()) +
                "\" desires have an invalid resource count (" + pair.first + "=" + to_string(pair.second) + ").");
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
            lock_guard<mutex> lock(m_pendingDesiresMutex);
            vector<unique_ptr<Desire>> desires;
            swap(m_pendingDesires, desires);
            updateStrategies(move(desires));
        }
    }
}

void HbbaLite::updateStrategies(vector<unique_ptr<Desire>> desires)
{
    auto enabledStrategies = m_solver->solve(desires, m_strategiesByDesireType, m_resourcesByNames);

    for (auto& p : m_strategiesByDesireType)
    {
        for (size_t i = 0; i < p.second.size(); i++)
        {
            bool toBeEnabled = enabledStrategies.count(pair<type_index, size_t>(p.first, i)) == 1;
            if (p.second[i]->enabled() && !toBeEnabled)
            {
                p.second[i]->disable();
            }
            else if (!p.second[i]->enabled() && toBeEnabled)
            {
                p.second[i]->enable();
            }
        }
    }
}
