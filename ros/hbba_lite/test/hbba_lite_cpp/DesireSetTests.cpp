#include <hbba_lite/DesireSet.h>

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

class DesireC : public Desire
{
public:
    DesireC() : Desire(1)
    {
    }

    ~DesireC() override = default;

    unique_ptr<Desire> clone() override
    {
        return make_unique<DesireC>(*this);
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
