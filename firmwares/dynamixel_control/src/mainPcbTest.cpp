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
    setupImu();

    // TODO setup a dynamixel
    pinMode(LIMIT_SWITCH_PIN, INPUT);
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
    constexpr size_t READ_COUNT = 10;
    constexpr uint32_t READ_DELAY_MS = 1000;

    DEBUG_SERIAL.println("--------------------Test Limit Switch------------------");
    for (size_t i = 0; i < READ_COUNT; i++)
    {
        DEBUG_SERIAL.print("Status: ");
        DEBUG_SERIAL.print(static_cast<int>(digitalRead(LIMIT_SWITCH_PIN)));
        delay(READ_DELAY_MS);
    }

}

static void testImu()
{
    constexpr size_t READ_COUNT = 10;
    constexpr uint32_t READ_DELAY_MS = 1000;

    DEBUG_SERIAL.println("------------------------Test IMU-----------------------");

    for (size_t i = 0; i < READ_COUNT; i++)
    {
        if (!imu.readData())
        {
            continue;
        }

        DEBUG_SERIAL.print("Acceleration X: ");
        DEBUG_SERIAL.print(imu.getAccelerationXInMPerSS());
        DEBUG_SERIAL.println(" m/(ss)");

        DEBUG_SERIAL.print("Acceleration Y: ");
        DEBUG_SERIAL.print(imu.getAccelerationYInMPerSS());
        DEBUG_SERIAL.println(" m/(ss)");

        DEBUG_SERIAL.print("Acceleration Z: ");
        DEBUG_SERIAL.print(imu.getAccelerationZInMPerSS());
        DEBUG_SERIAL.println(" m/(ss)");

        DEBUG_SERIAL.print("Angular Rate X: ");
        DEBUG_SERIAL.print(imu.getAngularRateXInRadPerS());
        DEBUG_SERIAL.println(" rad/s");

        DEBUG_SERIAL.print("Angular Rate Y: ");
        DEBUG_SERIAL.print(imu.getAngularRateYInRadPerS());
        DEBUG_SERIAL.println(" rad/s");

        DEBUG_SERIAL.print("Angular Rate Z: ");
        DEBUG_SERIAL.print(imu.getAngularRateZInRadPerS());
        DEBUG_SERIAL.println(" rad/s");

        DEBUG_SERIAL.print("Temperature: ");
        DEBUG_SERIAL.print(imu.getTemperatureInCelcius());
        DEBUG_SERIAL.println(" C");

        delay(READ_DELAY_MS);
    }

}

static void testDynamixel()
{
    // TODO
}

#endif
