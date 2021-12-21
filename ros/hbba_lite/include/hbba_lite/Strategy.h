#ifndef HBBA_LITE_STRATEGY_H
#define HBBA_LITE_STRATEGY_H

#include <hbba_lite/ClassMacros.h>

#include <cstdint>
#include <unordered_map>
#include <typeindex>
#include <string>

class FilterConfiguration
{
    uint16_t m_rate;
    bool m_hasRate;

public:
    FilterConfiguration();
    FilterConfiguration(uint16_t rate);

    uint16_t rate() const;
    bool hasRate() const;

    friend bool operator==(const FilterConfiguration& a, const FilterConfiguration& b);
    friend bool operator!=(const FilterConfiguration& a, const FilterConfiguration& b);
};

inline uint16_t FilterConfiguration::rate() const
{
    return m_rate;
}

inline bool FilterConfiguration::hasRate() const
{
    return m_hasRate;
}

bool operator==(const FilterConfiguration& a, const FilterConfiguration& b)
{
    if (a.m_hasRate && b.m_hasRate)
    {
        return a.m_rate == b.m_rate;
    }
    else
    {
        return a.m_hasRate == b.m_hasRate;
    }
}

bool operator!=(const FilterConfiguration& a, const FilterConfiguration& b)
{
    return !(a == b);
}

class Strategy
{
    bool m_enabled;
    uint16_t m_utility;
    std::unordered_map<std::string, uint16_t> m_ressourcesByName;
    std::unordered_map<std::string, FilterConfiguration> m_filterConfigurationByName;

public:
    Strategy(uint16_t utility,
        std::unordered_map<std::string, uint16_t> ressourcesByName,
        std::unordered_map<std::string, FilterConfiguration> filterConfigurationByName);
    virtual ~Strategy() = default;

    DECLARE_NOT_COPYABLE(Strategy);
    DECLARE_NOT_MOVABLE(Strategy);

    void enable();
    void disable();
    bool enabled() const;

    const std::unordered_map<std::string, uint16_t>& ressourcesByName() const;
    const std::unordered_map<std::string, FilterConfiguration> filterConfigurationByName() const;

    virtual std::type_index desireType() = 0;

protected:
    virtual void onEnableChanged() = 0;
};

inline void Strategy::enable()
{
    if (!m_enabled)
    {
        m_enabled = true;
        onEnableChanged();
    }
}

inline void Strategy::disable()
{
    if (m_enabled)
    {
        m_enabled = false;
        onEnableChanged();
    }
}

inline bool Strategy::enabled() const
{
    return m_enabled;
}

inline const std::unordered_map<std::string, uint16_t>& Strategy::ressourcesByName() const
{
    return m_ressourcesByName;
}

inline const std::unordered_map<std::string, FilterConfiguration> Strategy::filterConfigurationByName() const
{
    return m_filterConfigurationByName;
}

#endif
