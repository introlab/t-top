#include <hbba_lite/Desire.h>

#include <gtest/gtest.h>

using namespace std;

class DesireSet
{
public:
    static void enable(Desire& desire)
    {
        desire.enable();
    }

    static void disable(Desire& desire)
    {
        desire.disable();
    }
};

class DesireA : public Desire
{
public:
    DesireA(uint16_t intensity) : Desire(intensity)
    {
    }

    ~DesireA() override = default;

    unique_ptr<Desire> clone() override
    {
        return make_unique<DesireA>(*this);
    }

    type_index type() override
    {
        return type_index(typeid(*this));
    }
};

class DesireB : public Desire
{
public:
    DesireB(uint16_t intensity) : Desire(intensity)
    {
    }

    ~DesireB() override = default;

    unique_ptr<Desire> clone() override
    {
        return make_unique<DesireB>(*this);
    }

    type_index type() override
    {
        return type_index(typeid(*this));
    }
};

TEST(DesireTests, getters_shouldReturnTheRightValues)
{
    DesireA desireA(1);
    DesireB desireB(2);

    EXPECT_EQ(desireA.id(), 0);
    EXPECT_EQ(desireA.intensity(), 1);
    EXPECT_TRUE(desireA.enabled());

    EXPECT_EQ(desireB.id(), 1);
    EXPECT_EQ(desireB.intensity(), 2);
    EXPECT_TRUE(desireB.enabled());


    DesireSet::disable(desireA);
    EXPECT_FALSE(desireA.enabled());
    DesireSet::enable(desireA);
    EXPECT_TRUE(desireA.enabled());

    EXPECT_NE(typeid(DesireA), typeid(desireB));
}

TEST(DesireTests, clone_shouldCloneTheDesire)
{
    DesireA desire(1);
    DesireSet::disable(desire);

    auto clonedDesire = desire.clone();

    EXPECT_EQ(desire.id(), clonedDesire->id());
    EXPECT_EQ(desire.intensity(), clonedDesire->intensity());
    EXPECT_EQ(desire.enabled(), clonedDesire->enabled());
}
