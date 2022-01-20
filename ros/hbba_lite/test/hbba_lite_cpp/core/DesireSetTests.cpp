#include <hbba_lite/core/DesireSet.h>

#include "Desires.h"

#include <gtest/gtest.h>

using namespace std;

class DesireSetObserverMock : public DesireSetObserver
{
public:
    vector<vector<unique_ptr<Desire>>> desires;

    void onDesireSetChanged(const vector<unique_ptr<Desire>>& enabledDesires) override
    {
        desires.emplace_back(vector<unique_ptr<Desire>>());
        for (auto& desire : enabledDesires)
        {
            desires[desires.size() - 1].emplace_back(desire->clone());
        }
    }
};

TEST(DesireSetTests, addObserverRemoveObserver_shouldAddAndRemoveTheObserver)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addObserver(&observer);
    testee.addDesire(desire.clone());

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire.id());

    testee.removeObserver(&observer);
    testee.removeDesire(desire.id());

    ASSERT_EQ(observer.desires.size(), 1);
}

TEST(DesireSetTests, addDesire_shouldAddTheDesireAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addObserver(&observer);
    testee.addDesire(desire.clone());

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire.id());
}

TEST(DesireSetTests, removeDesire_shouldRemoveTheDesireAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    testee.removeDesire(desire.id());

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 0);
}

TEST(DesireSetTests, removeDesire_invalidId_shouldDoNothing)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    testee.removeDesire(999999999);

    ASSERT_EQ(observer.desires.size(), 0);
}

TEST(DesireSetTests, removeDesires_shouldRemoveTheMatchingDesires)
{
    DesireSetObserverMock observer;
    DesireC desire1;
    DesireC desire2;
    DesireD desire3;
    DesireSet testee;

    testee.addDesire(desire1.clone());
    testee.addDesire(desire2.clone());
    testee.addDesire(desire3.clone());

    testee.addObserver(&observer);
    testee.removeDesires(type_index(typeid(DesireC)));

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire3.id());
}

TEST(DesireSetTests, contains_shouldIfTheDesireIsInTheSet)
{
    DesireC desire;
    DesireSet testee;
    EXPECT_FALSE(testee.contains(desire.id()));

    testee.addDesire(desire.clone());
    EXPECT_TRUE(testee.contains(desire.id()));

    testee.removeDesire(desire.id());
    EXPECT_FALSE(testee.contains(desire.id()));
}

TEST(DesireSetTests, clear_empty_shouldDoNothing)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addObserver(&observer);
    testee.clear();

    ASSERT_EQ(observer.desires.size(), 0);
}

TEST(DesireSetTests, clear_shouldRemoveAllDesiresAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    testee.clear();

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 0);
}

TEST(DesireSetTests, enableAllDesires_shouldEnableAllDesiresAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());
    testee.disableAllDesires();

    testee.addObserver(&observer);
    testee.enableAllDesires();

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire.id());
    EXPECT_TRUE(observer.desires[0][0]->enabled());
}

TEST(DesireSetTests, disableAllDesires_shouldDisableAllDesiresAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    testee.disableAllDesires();

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 0);
}

TEST(DesireSetTests, addDesire_transaction_shouldAddTheDesireAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addObserver(&observer);
    {
        auto transaction = testee.beginTransaction();
        testee.addDesire(desire.clone());

        ASSERT_EQ(observer.desires.size(), 0);
    }

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire.id());
}

TEST(DesireSetTests, removeDesire_transaction_shouldRemoveTheDesireAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    {
        auto transaction = testee.beginTransaction();
        testee.removeDesire(desire.id());

        ASSERT_EQ(observer.desires.size(), 0);
    }

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 0);
}

TEST(DesireSetTests, enableAllDesires_transaction_shouldEnableAllDesiresAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());
    testee.disableAllDesires();

    testee.addObserver(&observer);
    {
        auto transaction = testee.beginTransaction();
        testee.enableAllDesires();

        ASSERT_EQ(observer.desires.size(), 0);
    }


    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 1);
    EXPECT_EQ(observer.desires[0][0]->id(), desire.id());
    EXPECT_TRUE(observer.desires[0][0]->enabled());
}

TEST(DesireSetTests, disableAllDesires_transaction_shouldDisableAllDesiresAndCallTheObservers)
{
    DesireSetObserverMock observer;
    DesireC desire;
    DesireSet testee;

    testee.addDesire(desire.clone());

    testee.addObserver(&observer);
    {
        auto transaction = testee.beginTransaction();
        testee.disableAllDesires();

        ASSERT_EQ(observer.desires.size(), 0);
    }

    ASSERT_EQ(observer.desires.size(), 1);
    ASSERT_EQ(observer.desires[0].size(), 0);
}
