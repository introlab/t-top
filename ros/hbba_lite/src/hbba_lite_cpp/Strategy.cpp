#include <hbba_lite/Strategy.h>

#include <hbba_lite/HbbaLiteException.h>

using namespace std;

FilterConfiguration::FilterConfiguration() : m_type(FilterType::ON_OFF), m_rate(0)
{
}

FilterConfiguration::FilterConfiguration(uint16_t rate) : m_type(FilterType::THROTTLING), m_rate(rate)
{
    if (rate == 0)
    {
        throw HbbaLiteException("Invalid rate (rate=" + to_string(rate) + ")");
    }
}

void FilterPool::add(const string& name, FilterType type)
{
    lock_guard<recursive_mutex> lock(m_mutex);

    auto it = m_typesByName.find(name);
    if (it == m_typesByName.end())
    {
        m_typesByName[name] = type;
        m_countsByName[name] = 0;
    }
    else if (it->second != type)
    {
        throw HbbaLiteException("Not matching type for the filter (" + name + ")");
    }
}

void FilterPool::enable(const string& name, const FilterConfiguration& configuration)
{
    lock_guard<recursive_mutex> lock(m_mutex);

    auto it = m_countsByName.find(name);
    if (it == m_countsByName.end())
    {
        throw HbbaLiteException("Not existing filter (" + name + ")");
    }
    if (m_typesByName[name] != configuration.type())
    {
        throw HbbaLiteException("Not compatible filter configuration (" + name + ")");
    }

    if (it->second == 0)
    {
        applyEnabling(name, configuration);
        m_lastFilterConfigurationByName[name] = configuration;
    }
    else if (configuration != m_lastFilterConfigurationByName[name])
    {
        throw HbbaLiteException("Not compatible filter configuration (" + name + ")");
    }

    it->second++;
}

void FilterPool::disable(const string& name)
{
    lock_guard<recursive_mutex> lock(m_mutex);

    auto it = m_countsByName.find(name);
    if (it == m_countsByName.end())
    {
        throw HbbaLiteException("Not existing filter (" + name + ")");
    }

    it->second--;
    if (it->second == 0)
    {
        applyDisabling(name);
    }
}
