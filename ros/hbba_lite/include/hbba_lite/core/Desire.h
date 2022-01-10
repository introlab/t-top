#ifndef HBBA_LITE_CORE_DESIRE_H
#define HBBA_LITE_CORE_DESIRE_H

#include <cstdint>
#include <atomic>
#include <memory>
#include <typeindex>

class Desire
{
    static std::atomic_uint64_t m_idCounter;

    uint64_t m_id;
    uint16_t m_intensity;
    bool m_enabled;

public:
    Desire(uint16_t intensity);
    virtual ~Desire();

    uint64_t id() const;
    uint16_t intensity() const;
    bool enabled() const;

    virtual std::unique_ptr<Desire> clone() = 0;
    virtual std::type_index type() = 0;

    void enable();
    void disable();
};

inline uint64_t Desire::id() const
{
    return m_id;
}

inline uint16_t Desire::intensity() const
{
    return m_intensity;
}

inline bool Desire::enabled() const
{
    return m_enabled;
}

inline void Desire::enable()
{
    m_enabled = true;
}

inline void Desire::disable()
{
    m_enabled = false;
}

#endif
