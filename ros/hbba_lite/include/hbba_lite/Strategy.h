#ifndef HBBA_LITE_STRATEGY_H
#define HBBA_LITE_STRATEGY_H

#include <hbba_lite/ClassMacros.h>

#include <cstdint>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <mutex>
#include <memory>

enum class FilterType
{
    ON_OFF, THROTTLING
};

class FilterConfiguration
{
    FilterType m_type;
    uint16_t m_rate;

public:
    FilterConfiguration();
    FilterConfiguration(uint16_t rate);

    FilterType type() const;
    uint16_t rate() const;

    friend bool operator==(const FilterConfiguration& a, const FilterConfiguration& b);
    friend bool operator!=(const FilterConfiguration& a, const FilterConfiguration& b);
};

inline FilterType FilterConfiguration::type() const
{
    return m_type;
}

inline uint16_t FilterConfiguration::rate() const
{
    return m_rate;
}

inline bool operator==(const FilterConfiguration& a, const FilterConfiguration& b)
{
    if (a.m_type == FilterType::THROTTLING && b.m_type == FilterType::THROTTLING)
    {
        return a.m_rate == b.m_rate;
    }
    else
    {
        return a.m_type == b.m_type;
    }
}

inline bool operator!=(const FilterConfiguration& a, const FilterConfiguration& b)
{
    return !(a == b);
}

class FilterPool
{

protected:
    std::unordered_map<std::string, FilterType> m_typesByName;
    std::unordered_map<std::string, int> m_countsByName;
    std::unordered_map<std::string, FilterConfiguration> m_lastFilterConfigurationByName;

    std::recursive_mutex m_mutex;

public:
    FilterPool() = default;

    DECLARE_NOT_COPYABLE(FilterPool);
    DECLARE_NOT_MOVABLE(FilterPool);

    virtual void add(const std::string& name, FilterType type);
    void enable(const std::string& name, const FilterConfiguration& configuration);
    void disable(const std::string& name);

protected:
    virtual void applyEnabling(const std::string& name, const FilterConfiguration& configuration) = 0;
    virtual void applyDisabling(const std::string& name) = 0;
};

template <class T>
class Strategy
{
    bool m_enabled;
    uint16_t m_utility;
    std::unordered_map<std::string, uint16_t> m_ressourcesByName;
    std::unordered_map<std::string, FilterConfiguration> m_filterConfigurationsByName;
    std::shared_ptr<FilterPool> m_filterPool;

public:
    Strategy(uint16_t utility,
        std::unordered_map<std::string, uint16_t> ressourcesByName,
        std::unordered_map<std::string, FilterConfiguration> filterConfigurationByName,
        std::shared_ptr<FilterPool> filterPool);
    virtual ~Strategy() = default;

    DECLARE_NOT_COPYABLE(Strategy);
    DECLARE_NOT_MOVABLE(Strategy);

    void enable();
    void disable();
    bool enabled() const;

    const std::unordered_map<std::string, uint16_t>& ressourcesByName() const;
    const std::unordered_map<std::string, FilterConfiguration> filterConfigurationsByName() const;

    std::type_index desireType();

protected:
    virtual void onEnabling();
    virtual void onDisabling();
};

template <class T>
inline Strategy<T>::Strategy(uint16_t utility,
    std::unordered_map<std::string, uint16_t> ressourcesByName,
    std::unordered_map<std::string, FilterConfiguration> filterConfigurationsByName,
    std::shared_ptr<FilterPool> filterPool) :
        m_enabled(false),
        m_utility(utility),
        m_ressourcesByName(move(ressourcesByName)),
        m_filterConfigurationsByName(move(filterConfigurationsByName)),
        m_filterPool(filterPool)
{
    for (auto& pair : m_filterConfigurationsByName)
    {
        m_filterPool->add(pair.first, pair.second.type());
    }
}

template <class T>
inline void Strategy<T>::enable()
{
    if (!m_enabled)
    {
        m_enabled = true;
        onEnabling();
    }
}

template <class T>
inline void Strategy<T>::disable()
{
    if (m_enabled)
    {
        m_enabled = false;
        onDisabling();
    }
}

template <class T>
inline bool Strategy<T>::enabled() const
{
    return m_enabled;
}

template <class T>
inline const std::unordered_map<std::string, uint16_t>& Strategy<T>::ressourcesByName() const
{
    return m_ressourcesByName;
}

template <class T>
inline const std::unordered_map<std::string, FilterConfiguration> Strategy<T>::filterConfigurationsByName() const
{
    return m_filterConfigurationsByName;
}

template <class T>
inline void Strategy<T>::onEnabling()
{
    for (auto& pair : m_filterConfigurationsByName)
    {
        m_filterPool->enable(pair.first, pair.second);
    }
}

template <class T>
inline void Strategy<T>::onDisabling()
{
    for (auto& pair : m_filterConfigurationsByName)
    {
        m_filterPool->disable(pair.first);
    }
}

template <class T>
inline std::type_index Strategy<T>::desireType()
{
    return std::type_index(typeid(T));
}

#endif
