#ifndef HBBA_LITE_UTILS_BINARY_SEMAPHORE_H
#define HBBA_LITE_UTILS_BINARY_SEMAPHORE_H

#include <mutex>
#include <condition_variable>

class BinarySemaphore
{
    bool m_state;
    std::mutex m_mutex;
    std::condition_variable m_conditionVariable;

public:
    BinarySemaphore(bool state);

    void release();
    void acquire();
    bool tryAcquire();

    template<class Rep, class Period>
    bool tryAcquireFor(const std::chrono::duration<Rep, Period>& duration);
};

inline void BinarySemaphore::release()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state = true;
    m_conditionVariable.notify_one();
}

inline void BinarySemaphore::acquire()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_conditionVariable.wait(lock, [this]() { return m_state; });
    m_state = false;
}

inline bool BinarySemaphore::tryAcquire()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_state)
    {
        m_state = false;
        return true;
    }
    else
    {
        return false;
    }
}

template<class Rep, class Period>
inline bool BinarySemaphore::tryAcquireFor(const std::chrono::duration<Rep, Period>& duration)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_conditionVariable.wait_for(lock, duration, [this]() { return m_state; }))
    {
        m_state = false;
        return true;
    }
    else
    {
        return false;
    }
}

#endif
