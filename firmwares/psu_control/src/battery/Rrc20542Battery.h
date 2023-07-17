#ifndef PSU_CONTROL_BATTERY_RRC_20542_BATTERY_H
#define PSU_CONTROL_BATTERY_RRC_20542_BATTERY_H

#include "battery/RrcUtils.h"

#include <ClassMacro.h>

#include <Arduino.h>
#include <Wire.h>

constexpr uint8_t RRC_20542_BATTERY_ADDRESS = 0x0B;

enum class RrcBatteryErrorCode
{
    OK = 0,
    BUSY = 1,
    RESERVED_COMMAND = 2,
    UNSUPPORTED_COMMAND = 3,
    ACCESS_DENIED = 4,
    OVERFLOW_UNDERFLOW = 5,
    BAD_SIZE = 6,
    UNKNOWN = 7
};

class Rrc20542Battery
{
    TwoWire& m_wire;

public:
    explicit Rrc20542Battery(TwoWire& wire);

    DECLARE_NOT_COPYABLE(Rrc20542Battery);
    DECLARE_NOT_MOVABLE(Rrc20542Battery);

    bool readTemperature(
        float& temperature,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the temperature in degrees Celsius

    bool readVoltage(float& voltage, size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the voltage in V
    bool readCurrent(float& current, size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the current in A
    bool readAverageCurrent(
        float& current,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the 1 minute average current over in A

    bool readRelativeStateOfCharge(
        float& stateOfCharge,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the relative state of charge in percents
    bool readAbsoluteStateOfCharge(
        float& stateOfCharge,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the absolute state of charge in percents
    bool readRemainingCapacity(
        float& capacity,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the remaining capacity in Ah
    bool readFullChargeCapacity(
        float& capacity,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the full charge capacity in Ah

    bool readRunTimeToEmpty(
        float& time,
        size_t trialCount =
            RRC_DEFAULT_TRIAL_COUNT);  // Read the remaining battery runtime at the current discharge rate in minutes
    bool readAverageTimeToEmpty(
        float& time,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the remaining battery runtime at the 1 minute average
                                                       // discharge rate in minutes
    bool readAverageTimeToFull(
        float& time,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the remaining charging time in minutes

    bool readBatteryStatus(
        bool& isFullyDischarged,
        bool& isFullyCharged,
        RrcBatteryErrorCode& error,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the battery status
    bool readCycleCount(uint16_t& cycleCount, size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the cycle count

    bool readDesignCapacity(
        float& capacity,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the design capacity in Ah
    bool
        readDesignVoltage(float& voltage, size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the design voltage in V

    bool readManufacturerName(
        char* name,
        uint8_t maxNameSize,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the manufacturer name
    bool readDeviceName(
        char* name,
        uint8_t maxNameSize,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the device name
    bool readDeviceChemistry(
        char* name,
        uint8_t maxNameSize,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the device chemistry name
    bool readManufactureDate(
        int& day,
        int& month,
        int& year,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the manufacture date
    bool readSerialNumber(
        uint16_t& serialNumber,
        size_t trialCount = RRC_DEFAULT_TRIAL_COUNT);  // Read the serial number

private:
    bool readWord(uint8_t command, uint16_t& value);
    bool readBlock(uint8_t command, uint8_t maxSize, char* data, uint8_t& size);

    bool readWordTrials(uint8_t command, uint16_t& value, size_t trialCount);
    bool readBlockTrials(uint8_t command, uint8_t maxSize, char* data, uint8_t& size, size_t trialCount);
};

#endif
