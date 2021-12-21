#ifndef HBBA_LITE_MOTIVATION_H
#define HBBA_LITE_MOTIVATION_H

#include <hbba_lite/DesireSet.h>

#include <memory>
#include <atomic>
#include <thread>

class Motivation
{
protected:
    std::shared_ptr<DesireSet> m_desireSet;

public:
    Motivation(std::shared_ptr<DesireSet> desireSet);
    virtual ~Motivation() = default;
};

class ThreadedMotivation : public Motivation
{
    std::atomic_bool m_stopped;
    std::unique_ptr<std::thread> m_thread;

public:
    ThreadedMotivation(std::shared_ptr<DesireSet> desireSet);
    ~ThreadedMotivation() override;

    void start();

protected:
    bool stopped();

    virtual void run() = 0;
};

inline bool ThreadedMotivation::stopped()
{
    return m_stopped.load();
}

#endif
