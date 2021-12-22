#ifndef FILTER_POOL_MOCK
#define FILTER_POOL_MOCK

#include <hbba_lite/core/Strategy.h>

class FilterPoolMock : public FilterPool
{
public:
    std::unordered_map<std::string, FilterConfiguration> enabledFilters;
    std::unordered_map<std::string, int> counts;

    void add(const std::string& name, FilterType type) override
    {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        FilterPool::add(name, type);
        counts[name] = 0;
    }

protected:
    void applyEnabling(const std::string& name, const FilterConfiguration& configuration) override
    {
        enabledFilters[name] = configuration;
        counts[name]++;
    }

    void applyDisabling(const std::string& name) override
    {
        enabledFilters.erase(name);
        counts[name]--;
    }
};

#endif
