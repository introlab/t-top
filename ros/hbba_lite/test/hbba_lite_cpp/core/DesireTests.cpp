#include <hbba_lite/core/Desire.h>

#include "Desires.h"

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
