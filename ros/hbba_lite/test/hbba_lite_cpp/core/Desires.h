#ifndef DESIRES_H
#define DESIRES_H

#include <hbba_lite/core/Desire.h>

class DesireA : public Desire
{
public:
    DesireA(uint16_t intensity) : Desire(intensity)
    {
    }

    ~DesireA() override = default;

    std::unique_ptr<Desire> clone() override
    {
        return std::make_unique<DesireA>(*this);
    }

    std::type_index type() override
    {
        return std::type_index(typeid(*this));
    }
};

class DesireB : public Desire
{
public:
    DesireB(uint16_t intensity) : Desire(intensity)
    {
    }

    ~DesireB() override = default;

    std::unique_ptr<Desire> clone() override
    {
        return std::make_unique<DesireB>(*this);
    }

    std::type_index type() override
    {
        return std::type_index(typeid(*this));
    }
};

class DesireC : public Desire
{
public:
    DesireC() : Desire(1)
    {
    }

    ~DesireC() override = default;

    std::unique_ptr<Desire> clone() override
    {
        return std::make_unique<DesireC>(*this);
    }

    std::type_index type() override
    {
        return std::type_index(typeid(*this));
    }
};

class DesireD : public Desire
{
public:
    DesireD() : Desire(1)
    {
    }

    ~DesireD() override = default;

    std::unique_ptr<Desire> clone() override
    {
        return std::make_unique<DesireD>(*this);
    }

    std::type_index type() override
    {
        return std::type_index(typeid(*this));
    }
};

#endif
