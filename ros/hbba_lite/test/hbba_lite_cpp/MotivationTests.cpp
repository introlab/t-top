#include <hbba_lite/Motivation.h>

#include <gtest/gtest.h>

using namespace std;

class ThreadedMotivationTestee : public ThreadedMotivation
{
public:
    bool& finished;

    ThreadedMotivationTestee(std::shared_ptr<DesireSet> desireSet, bool& finished) :
            ThreadedMotivation(desireSet),
            finished(finished)
    {
    }

    ~ThreadedMotivationTestee() override = default;

protected:
    void run() override
    {
        while (!stopped())
        {
        }
        finished = true;
    }
};

TEST(ThreadedMotivationTests, constructor_shouldStartAThread)
{
    bool finished = false;

    {
        ThreadedMotivationTestee testee(make_shared<DesireSet>(), finished);
        testee.start();
    }

    EXPECT_TRUE(finished);
}
