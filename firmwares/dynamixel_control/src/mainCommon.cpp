#include "mainCommon.h"

Icm40627 imu(Wire, IMU_INT1_PIN, IMU_INT2_PIN, IMU_ADDRESS);
Dynamixel2Arduino dynamixel(DYNAMIXEL_SERIAL, DYNAMIXEL_DIR_PIN);

void setupDebugSerial()
{
    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);
}

void setupWire()
{
    DEBUG_SERIAL.println("Setup Wire - Start");
    Wire.setClock(WIRE_CLOCK);
    Wire.begin();
    DEBUG_SERIAL.println("Setup Wire - End");
}

void setupImu()
{
    DEBUG_SERIAL.println("Setup Imu - Start");
    if (!imu.begin(IMU_ACCELEROMETER_RANGE, IMU_GYROSCOPE_RANGE, IMU_ODR, IMU_ANTI_ALIAS_FILTER_BANDWIDTH))
    {
        CRITICAL_ERROR("Setup Imu - failure");
    }
    DEBUG_SERIAL.println("Setup Imu - End");
}

void setupImu(void (*dataReadyInterrupt)())
{
    DEBUG_SERIAL.println("Setup Imu - Start");
    if (!imu.begin(IMU_ACCELEROMETER_RANGE, IMU_GYROSCOPE_RANGE, IMU_ODR, IMU_ANTI_ALIAS_FILTER_BANDWIDTH, dataReadyInterrupt))
    {
        CRITICAL_ERROR("Setup Imu - failure");
    }
    DEBUG_SERIAL.println("Setup Imu - End");
}

void setupDynamixel()
{
    DEBUG_SERIAL.println("Setup Dynamixel - Start");

    pinMode(DYNAMIXEL_ENABLE_PIN, OUTPUT);
    digitalWrite(DYNAMIXEL_ENABLE_PIN, true);

    dynamixel.begin(DYNAMIXEL_BAUD_RATE);

    DEBUG_SERIAL.println("Setup Dynamixel - End");
}
