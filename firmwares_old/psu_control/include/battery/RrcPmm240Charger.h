#ifndef RRC_PMM_240_CHARGER_H
#define RRC_PMM_240_CHARGER_H

#include "battery/RrcUtils.h"

#include <Arduino.h>
#include <Wire.h>

constexpr uint8_t RRC_PMM_240_CHARGER_ADDRESS = 0x10;

class RrcPmm240Charger
{
    enum class RrcPmm240ChargerMemoryLocation : uint8_t
    {
        RAM = 0,
        EEPROM = 1
    };
    TwoWire& m_wire;
    uint8_t m_batteryStatusPin;
    uint8_t m_chargerStatusPin;

    volatile uint32_t m_lastBatteryStatusInterruptMs;
    volatile uint32_t m_currentBatteryStatusInterruptMs;
    volatile uint32_t m_lastChargerStatusInterruptMs;
    volatile uint32_t m_currentChargerStatusInterruptMs;

public:
    RrcPmm240Charger(TwoWire& wire, uint8_t batteryStatusPin, uint8_t chargerStatusPin);
    void begin();

    bool writeChargeCurrentLimitToRam(
        float current,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Write the charge current limit to RAM. The current unit is A.
    bool writeChargeCurrentLimitToEeprom(
        float current,
        size_t trialCount =
            RRC_DEFAULT_TRIAL_COUNT);  // Write the charge current limit to EEPROM. The current unit is A.

    bool writeInputCurrentLimitToRam(
        float current,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Write the input current limit to RAM. The current unit is A.
    bool writeInputCurrentLimitToEeprom(
        float current,
        size_t trialCount =
            RRC_DEFAULT_TRIAL_COUNT);  // Write the input current limit to EEPROM. The current unit is A.

    bool isBatteryCharged();
    bool isBatteryCharging();
    bool hasBatteryError();

    bool isPsuConnected();
    bool hasChargerError();

private:
    bool writeChargeCurrentLimit(float current, RrcPmm240ChargerMemoryLocation location, size_t trialCount);
    bool writeInputCurrentLimit(float current, RrcPmm240ChargerMemoryLocation location, size_t trialCount);

    bool writeWord(uint8_t command, RrcPmm240ChargerMemoryLocation location, uint16_t value);
    bool writeWordTrials(uint8_t command, RrcPmm240ChargerMemoryLocation location, uint16_t value, size_t trialCount);
    uint8_t crc8(uint8_t* data, size_t size);
};

#endif
