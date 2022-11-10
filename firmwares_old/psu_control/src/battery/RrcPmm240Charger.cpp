#include "battery/RrcPmm240Charger.h"
#include "utils/InterruptLock.h"

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

uint8_t RrcPmm240Charger::crc8(uint8_t* data, size_t size)
{
    // Inspired by https://stackoverflow.com/a/21782081 and http://www.sunshine2k.de/coding/javascript/crc/crc_js.html
    constexpr uint8_t LOOKUP_TABLE[] = {
        0x00, 0x07, 0x0e, 0x09, 0x1c, 0x1b, 0x12, 0x15, 0x38, 0x3f, 0x36, 0x31, 0x24, 0x23, 0x2a, 0x2d, 0x70, 0x77,
        0x7e, 0x79, 0x6c, 0x6b, 0x62, 0x65, 0x48, 0x4f, 0x46, 0x41, 0x54, 0x53, 0x5a, 0x5d, 0xe0, 0xe7, 0xee, 0xe9,
        0xfc, 0xfb, 0xf2, 0xf5, 0xd8, 0xdf, 0xd6, 0xd1, 0xc4, 0xc3, 0xca, 0xcd, 0x90, 0x97, 0x9e, 0x99, 0x8c, 0x8b,
        0x82, 0x85, 0xa8, 0xaf, 0xa6, 0xa1, 0xb4, 0xb3, 0xba, 0xbd, 0xc7, 0xc0, 0xc9, 0xce, 0xdb, 0xdc, 0xd5, 0xd2,
        0xff, 0xf8, 0xf1, 0xf6, 0xe3, 0xe4, 0xed, 0xea, 0xb7, 0xb0, 0xb9, 0xbe, 0xab, 0xac, 0xa5, 0xa2, 0x8f, 0x88,
        0x81, 0x86, 0x93, 0x94, 0x9d, 0x9a, 0x27, 0x20, 0x29, 0x2e, 0x3b, 0x3c, 0x35, 0x32, 0x1f, 0x18, 0x11, 0x16,
        0x03, 0x04, 0x0d, 0x0a, 0x57, 0x50, 0x59, 0x5e, 0x4b, 0x4c, 0x45, 0x42, 0x6f, 0x68, 0x61, 0x66, 0x73, 0x74,
        0x7d, 0x7a, 0x89, 0x8e, 0x87, 0x80, 0x95, 0x92, 0x9b, 0x9c, 0xb1, 0xb6, 0xbf, 0xb8, 0xad, 0xaa, 0xa3, 0xa4,
        0xf9, 0xfe, 0xf7, 0xf0, 0xe5, 0xe2, 0xeb, 0xec, 0xc1, 0xc6, 0xcf, 0xc8, 0xdd, 0xda, 0xd3, 0xd4, 0x69, 0x6e,
        0x67, 0x60, 0x75, 0x72, 0x7b, 0x7c, 0x51, 0x56, 0x5f, 0x58, 0x4d, 0x4a, 0x43, 0x44, 0x19, 0x1e, 0x17, 0x10,
        0x05, 0x02, 0x0b, 0x0c, 0x21, 0x26, 0x2f, 0x28, 0x3d, 0x3a, 0x33, 0x34, 0x4e, 0x49, 0x40, 0x47, 0x52, 0x55,
        0x5c, 0x5b, 0x76, 0x71, 0x78, 0x7f, 0x6a, 0x6d, 0x64, 0x63, 0x3e, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2c, 0x2b,
        0x06, 0x01, 0x08, 0x0f, 0x1a, 0x1d, 0x14, 0x13, 0xae, 0xa9, 0xa0, 0xa7, 0xb2, 0xb5, 0xbc, 0xbb, 0x96, 0x91,
        0x98, 0x9f, 0x8a, 0x8d, 0x84, 0x83, 0xde, 0xd9, 0xd0, 0xd7, 0xc2, 0xc5, 0xcc, 0xcb, 0xe6, 0xe1, 0xe8, 0xef,
        0xfa, 0xfd, 0xf4, 0xf3};
    uint8_t crc8Value = 0;

    for (size_t i = 0; i < size; i++)
    {
        crc8Value = LOOKUP_TABLE[(crc8Value ^ data[i])] ^ (crc8Value << 8);
    }

    return crc8Value;
}
