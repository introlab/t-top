#include "RrcPmm240Charger.h"
#include "../utils/InterruptLock.h"

#include <Crc8.h>

constexpr uint32_t BATTERY_BLINK_TIMEOUT_MS = 3000;
constexpr int64_t BATTERY_CHARGING_PERIOD_MS = 2000;
constexpr int64_t BATTERY_ERROR_PERIOD_MS = 400;
constexpr uint32_t CHARGER_BLINK_TIMEOUT_MS = 500;

static volatile uint32_t* lastBatteryStatusInterruptMs;
static volatile uint32_t* currentBatteryStatusInterruptMs;

static volatile uint32_t* lastChargerStatusInterruptMs;
static volatile uint32_t* currentChargerStatusInterruptMs;

static void batteryStatusInterrupt()
{
    *lastBatteryStatusInterruptMs = *currentBatteryStatusInterruptMs;
    *currentBatteryStatusInterruptMs = millis();
}

static void chargerStatusInterrupt()
{
    *lastChargerStatusInterruptMs = *currentChargerStatusInterruptMs;
    *currentChargerStatusInterruptMs = millis();
}

RrcPmm240Charger::RrcPmm240Charger(TwoWire& wire, uint8_t batteryStatusPin, uint8_t chargerStatusPin)
    : m_wire(wire),
      m_batteryStatusPin(batteryStatusPin),
      m_chargerStatusPin(chargerStatusPin)
{
}

void RrcPmm240Charger::begin()
{
    m_lastBatteryStatusInterruptMs = 0;
    m_currentBatteryStatusInterruptMs = 0;
    m_lastChargerStatusInterruptMs = 0;
    m_currentChargerStatusInterruptMs = 0;

    lastBatteryStatusInterruptMs = &m_lastBatteryStatusInterruptMs;
    currentBatteryStatusInterruptMs = &m_currentBatteryStatusInterruptMs;
    lastChargerStatusInterruptMs = &m_lastChargerStatusInterruptMs;
    currentChargerStatusInterruptMs = &m_currentChargerStatusInterruptMs;

    pinMode(m_batteryStatusPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(m_batteryStatusPin), batteryStatusInterrupt, RISING);

    pinMode(m_chargerStatusPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(m_chargerStatusPin), chargerStatusInterrupt, RISING);
}

bool RrcPmm240Charger::writeChargeCurrentLimitToRam(float current, size_t trialCount)
{
    return writeChargeCurrentLimit(current, RrcPmm240ChargerMemoryLocation::RAM, trialCount);
}

bool RrcPmm240Charger::writeChargeCurrentLimitToEeprom(float current, size_t trialCount)
{
    return writeChargeCurrentLimit(current, RrcPmm240ChargerMemoryLocation::EEPROM, trialCount);
}

bool RrcPmm240Charger::writeInputCurrentLimitToRam(float current, size_t trialCount)
{
    return writeInputCurrentLimit(current, RrcPmm240ChargerMemoryLocation::RAM, trialCount);
}

bool RrcPmm240Charger::writeInputCurrentLimitToEeprom(float current, size_t trialCount)
{
    return writeInputCurrentLimit(current, RrcPmm240ChargerMemoryLocation::EEPROM, trialCount);
}

bool RrcPmm240Charger::isBatteryCharged()
{
    PinInterruptLock lock;

    if ((millis() - m_currentBatteryStatusInterruptMs) < BATTERY_BLINK_TIMEOUT_MS)
    {
        return false;
    }
    return digitalRead(m_batteryStatusPin);
}

bool RrcPmm240Charger::isBatteryCharging()
{
    PinInterruptLock lock;

    if ((millis() - m_currentBatteryStatusInterruptMs) >= BATTERY_BLINK_TIMEOUT_MS)
    {
        return false;
    }

    int64_t period = m_currentBatteryStatusInterruptMs - m_lastBatteryStatusInterruptMs;
    return abs(BATTERY_CHARGING_PERIOD_MS - period) < abs(BATTERY_ERROR_PERIOD_MS - period);
}

bool RrcPmm240Charger::hasBatteryError()
{
    PinInterruptLock lock;

    if ((millis() - m_currentBatteryStatusInterruptMs) >= BATTERY_BLINK_TIMEOUT_MS)
    {
        return false;
    }

    int64_t period = m_currentBatteryStatusInterruptMs - m_lastBatteryStatusInterruptMs;
    return abs(BATTERY_ERROR_PERIOD_MS - period) < abs(BATTERY_CHARGING_PERIOD_MS - period);
}

bool RrcPmm240Charger::isPsuConnected()
{
    PinInterruptLock lock;

    if ((millis() - m_currentChargerStatusInterruptMs) < CHARGER_BLINK_TIMEOUT_MS)
    {
        return false;
    }
    return digitalRead(m_chargerStatusPin);
}

bool RrcPmm240Charger::hasChargerError()
{
    PinInterruptLock lock;
    return (millis() - m_currentChargerStatusInterruptMs) < CHARGER_BLINK_TIMEOUT_MS;
}

bool RrcPmm240Charger::writeChargeCurrentLimit(
    float current,
    RrcPmm240ChargerMemoryLocation location,
    size_t trialCount)
{
    if (current < 0.256f || current > 6.2f)
    {
        return false;
    }
    uint16_t currentMa = static_cast<uint16_t>(current * 1000);
    return writeWordTrials(0x3c, location, currentMa, trialCount);
}

bool RrcPmm240Charger::writeInputCurrentLimit(float current, RrcPmm240ChargerMemoryLocation location, size_t trialCount)
{
    if (current < 0.0f || current > 16.128f)
    {
        return false;
    }
    uint16_t currentMa = static_cast<uint16_t>(current * 1000);
    return writeWordTrials(0x3d, location, currentMa, trialCount);
}

bool RrcPmm240Charger::writeWord(uint8_t command, RrcPmm240ChargerMemoryLocation location, uint16_t value)
{
    RrcWordUnion word;
    word.word = value;

    uint8_t data[] =
        {RRC_PMM_240_CHARGER_ADDRESS << 1, command, 0x03, static_cast<uint8_t>(location), word.bytes[1], word.bytes[0]};
    uint8_t pec = crc8(data, sizeof(data));

    m_wire.beginTransmission(RRC_PMM_240_CHARGER_ADDRESS);
    for (size_t i = 1; i < sizeof(data); i++)
    {
        m_wire.write(data[i]);
    }
    m_wire.write(pec);

    return m_wire.endTransmission() == 0;
}

bool RrcPmm240Charger::writeWordTrials(
    uint8_t command,
    RrcPmm240ChargerMemoryLocation location,
    uint16_t value,
    size_t trialCount)
{
    for (size_t i = 0; i < trialCount; i++)
    {
        if (writeWord(command, location, value))
        {
            return true;
        }
        smBusRandomDelay();
    }
    return false;
}
