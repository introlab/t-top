#include <Dynamixel2Arduino.h>


#define DEBUG_SERIAL Serial
const long DEBUG_SERIAL_BAUD_RATE = 9600;

#define DYNAMIXEL_SERIAL Serial2
constexpr uint8_t DYNAMIXEL_DIR_PIN = 9;
constexpr uint8_t DYNAMIXEL_ENABLE_PIN = 2;
constexpr float DYNAMIXEL_PROTOCOL_VERSION = 2.0;

constexpr uint8_t MOTOR_ID = 1;
constexpr uint8_t NEW_MOTOR_ID = 1;  // TODO change

constexpr unsigned long MOTOR_BAUD_RATE = 57600;
constexpr unsigned long NEW_MOTOR_BAUD_RATE = 4000000;

constexpr uint8_t RETURN_DELAY_TIME = 5;

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
    if (!dynamixel.writeControlTableItem(ControlTableItem::RETURN_DELAY_TIME, MOTOR_ID, RETURN_DELAY_TIME))
    {
        DEBUG_SERIAL.println("setReturnDelayTime failed");
        return;
    }
    DEBUG_SERIAL.println("setReturnDelayTime succeeded");

    if (!dynamixel.setID(MOTOR_ID, NEW_MOTOR_ID))
    {
        DEBUG_SERIAL.println("setID failed");
        return;
    }
    DEBUG_SERIAL.println("setID succeeded");

    if (!dynamixel.setBaudrate(NEW_MOTOR_ID, NEW_MOTOR_BAUD_RATE))
    {
        DEBUG_SERIAL.println("setBaudrate failed");
        return;
    }
    DEBUG_SERIAL.println("setBaudrate succeeded");
}

void loop() {}
