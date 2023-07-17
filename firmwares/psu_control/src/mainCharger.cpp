#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_SETUP_BATTERY_CHARGER

#include "mainCommon.h"

constexpr uint32_t STARTUP_DELAY = 5000;
constexpr uint32_t COMMAND_DELAY = 100;
constexpr uint32_t STATUS_DELAY = 10000;

void setChargerCurrentLimit();

void setup()
{
    delay(STARTUP_DELAY);
    initRandom();

    setupDebugSerial();
    setupWire();
    setupCharger();

    setChargerCurrentLimit();
}

void loop() {}

void setChargerCurrentLimit()
{
    DEBUG_SERIAL.println("Set Charger Current Limit - Start");

    if (!charger.writeChargeCurrentLimitToEeprom(BATTERY_CHARGER_CHARGE_CURRENT_LIMIT))
    {
        DEBUG_SERIAL.println("writeChargeCurrentLimitToEeprom failed");
    }
    delay(COMMAND_DELAY);

    if (!charger.writeChargeCurrentLimitToRam(BATTERY_CHARGER_CHARGE_CURRENT_LIMIT))
    {
        DEBUG_SERIAL.println("writeChargeCurrentLimitToRam failed");
    }
    delay(COMMAND_DELAY);

    if (!charger.writeInputCurrentLimitToEeprom(BATTERY_CHARGER_INPUT_CURRENT_LIMIT))
    {
        DEBUG_SERIAL.println("writeInputCurrentLimitToEeprom failed");
    }
    delay(COMMAND_DELAY);

    if (!charger.writeInputCurrentLimitToRam(BATTERY_CHARGER_INPUT_CURRENT_LIMIT))
    {
        DEBUG_SERIAL.println("writeInputCurrentLimitToRam failed");
    }

    DEBUG_SERIAL.println("Set Charger Current Limit - End");
}

#endif
