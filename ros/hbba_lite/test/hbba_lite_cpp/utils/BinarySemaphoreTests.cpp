#include <hbba_lite/utils/BinarySemaphore.h>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace std;

TEST(BinarySemaphoreTests, releaseAcquire_shouldReleaseAcquireTheSemaphore)
{
    BinarySemaphore semaphore(false);
    bool finished = false;

    thread t([&]()
    {
        semaphore.acquire();
        finished = true;
    });

    this_thread::sleep_for(10ms);
    semaphore.release();

    t.join();

    EXPECT_TRUE(finished);
}

TEST(BinarySemaphoreTests, tryAcquire_shouldTryAcquireTheSemaphore)
{
    BinarySemaphore semaphore(false);

    EXPECT_FALSE(semaphore.tryAcquire());
    semaphore.release();
    semaphore.release();
    EXPECT_TRUE(semaphore.tryAcquire());
    EXPECT_FALSE(semaphore.tryAcquire());
}

TEST(BinarySemaphoreTests, tryAcquireFor_shouldWaitAndReturn)
{
    BinarySemaphore semaphore(false);

    auto start = chrono::steady_clock::now();
    EXPECT_FALSE(semaphore.tryAcquireFor(10ms));
    auto end = chrono::steady_clock::now();
    auto duration = end - start;
    EXPECT_NEAR(chrono::duration_cast<chrono::milliseconds>(duration).count(), 10, 2);

    semaphore.release();

    start = chrono::steady_clock::now();
    EXPECT_TRUE(semaphore.tryAcquire());
    end = chrono::steady_clock::now();
    duration = end - start;
    EXPECT_NEAR(chrono::duration_cast<chrono::milliseconds>(duration).count(), 0, 2);
}
