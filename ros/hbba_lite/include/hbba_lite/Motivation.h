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

#endif
