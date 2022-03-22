#ifndef DESIRES_H
#define DESIRES_H

#include <hbba_lite/core/Desire.h>

class DesireA : public Desire
{
public:
    DesireA(uint16_t intensity) : Desire(intensity) {}

    ~DesireA() override = default;

    DECLARE_DESIRE_METHODS(DesireA)
};

class DesireB : public Desire
{
public:
    DesireB(uint16_t intensity) : Desire(intensity) {}

    ~DesireB() override = default;

    DECLARE_DESIRE_METHODS(DesireB)
};

class DesireC : public Desire
{
public:
    DesireC() : Desire(1) {}

    ~DesireC() override = default;

    DECLARE_DESIRE_METHODS(DesireC)
};

class DesireD : public Desire
{
public:
    DesireD() : Desire(1) {}

    ~DesireD() override = default;

    DECLARE_DESIRE_METHODS(DesireD)
};

#endif
