#include <Dynamixel2Arduino.h>


#define DEBUG_SERIAL Serial
const long DEBUG_SERIAL_BAUD_RATE = 9600;

#define DYNAMIXEL_SERIAL Serial2
constexpr uint8_t DYNAMIXEL_DIR_PIN = 9;
constexpr uint8_t DYNAMIXEL_ENABLE_PIN = 2;
constexpr float DYNAMIXEL_PROTOCOL_VERSION = 2.0;

constexpr uint8_t MOTOR_ID = 1;
constexpr unsigned long MOTOR_BAUD_RATE = 1000000;

constexpr uint32_t STARTUP_DELAY_MS = 5000;

static Dynamixel2Arduino dynamixel(DYNAMIXEL_SERIAL, DYNAMIXEL_DIR_PIN);

void setup()
{
    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);

    pinMode(DYNAMIXEL_ENABLE_PIN, OUTPUT);
    digitalWrite(DYNAMIXEL_ENABLE_PIN, true);

    delay(STARTUP_DELAY_MS);

    dynamixel.begin(MOTOR_BAUD_RATE);
    if (!dynamixel.setPortProtocolVersion(DYNAMIXEL_PROTOCOL_VERSION))
    {
        DEBUG_SERIAL.println("setPortProtocolVersion failed");
    }
    if (!dynamixel.ping(MOTOR_ID))
    {
        DEBUG_SERIAL.println("ping failed");
    }

    if (!dynamixel.torqueOff(MOTOR_ID))
    {
        DEBUG_SERIAL.println("torqueOff failed");
        return;
    }
    if (!dynamixel.setOperatingMode(MOTOR_ID, OP_POSITION))
    {
        DEBUG_SERIAL.println("setOperatingMode failed");
        return;
    }
    if (!dynamixel.torqueOn(MOTOR_ID))
    {
        DEBUG_SERIAL.println("torqueOn failed");
        return;
    }

    if (!dynamixel.setGoalPosition(MOTOR_ID, 0, UNIT_DEGREE))
    {
        DEBUG_SERIAL.println("setGoalPosition failed");
        return;
    }
}

void loop() {}
