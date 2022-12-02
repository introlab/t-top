#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_PCB_TEST

#include "mainCommon.h"

constexpr uint32_t TEST_DELAY_MS = 5000;

#define TEST(function)                                                                                                 \
    function();                                                                                                        \
    DEBUG_SERIAL.println();                                                                                            \
    delay(TEST_DELAY_MS)

void setup()
{
    // put your setup code here, to run once:
    setupDebugSerial();
    setupWire();

    // TODO
}

static void testLimitSwitch();
static void testImu();
static void testDynamixel();

void loop()
{
    TEST(testLimitSwitch);
    TEST(testImu);
    TEST(testDynamixel);
}

static void testLimitSwitch()
{
    // TODO
}

static void testImu()
{
    // TODO
}

static void testDynamixel()
{
    // TODO
}

#endif