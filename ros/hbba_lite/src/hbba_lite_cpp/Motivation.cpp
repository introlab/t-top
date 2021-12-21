#include <hbba_lite/Motivation.h>

using namespace std;

Motivation::Motivation(shared_ptr<DesireSet> desireSet) : m_desireSet(move(desireSet))
{
}

ThreadedMotivation::ThreadedMotivation(shared_ptr<DesireSet> desireSet) :
        Motivation(desireSet),
        m_stopped(false)
{
}

ThreadedMotivation::~ThreadedMotivation()
{
    if (m_thread)
    {
        m_stopped.store(true);
        m_thread->join();
    }
}

void ThreadedMotivation::start()
{
    m_thread = make_unique<thread>([this]() { run(); });
}
