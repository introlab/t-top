#include <hbba_lite/Strategy.h>

using namespace std;

FilterConfiguration::FilterConfiguration() : m_hasRate(false), m_rate(0)
{
}

FilterConfiguration::FilterConfiguration(uint16_t rate) : m_hasRate(true), m_rate(rate)
{
}

Strategy::Strategy(uint16_t utility,
    std::unordered_map<std::string, uint16_t> ressourcesByName,
    std::unordered_map<std::string, FilterConfiguration> filterConfigurationByName) :
        m_enabled(false),
        m_utility(utility),
        m_ressourcesByName(move(ressourcesByName)),
        m_filterConfigurationByName(move(filterConfigurationByName))
{
}
