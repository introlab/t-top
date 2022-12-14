#include "Icm40627.h"

#define CHECK_BOOL_ERROR(code) \
    do \
    {\
        if (!(code)) \
        { \
            return false; \
        } \
    } while (false)


static bool writeRegister(TwoWire& wire, uint8_t address, uint8_t registerAddress, uint8_t value)
{
    wire.beginTransmission(address);
    wire.write(registerAddress);
    wire.write(value);
    return wire.endTransmission() == 0;
}

static bool setRegisterBank(TwoWire& wire, uint8_t address, Icm40627::RegisterBank registerBank)
{
    constexpr uint8_t REGISTER_BANK_ADDRESS = 0x76;
    return writeRegister(wire, address, REGISTER_BANK_ADDRESS, static_cast<uint8_t>(registerBank));
}

Icm40627::Icm40627(TwoWire& wire, uint8_t int1Pin, uint8_t int2Pin, uint8_t address) :
     m_wire(wire),
     m_int1Pin(int1Pin),
     m_int2Pin(int2Pin),
     m_address(address),
     m_currentRegisterBank(RegisterBank::BANK_0),
    m_accelerometerRangeInG(0.f),
    m_gyroscopeRangeInDegPerS(0.f),
    m_accelerationX(0),
    m_accelerationY(0),
    m_accelerationZ(0),
    m_angularRateX(0),
    m_angularRateY(0),
    m_angularRateZ(0)
{

}

bool Icm40627::begin(AccelerometerRange accelerometerRange, GyroscopeRange gyroscopeRange, Odr odr, AntiAliasFilterBandwidth antiAliasFilterBandwidth)
{
    return begin(accelerometerRange, gyroscopeRange, odr, antiAliasFilterBandwidth, nullptr);
}

bool Icm40627::begin(AccelerometerRange accelerometerRange,
        GyroscopeRange gyroscopeRange,
        Odr odr,
        AntiAliasFilterBandwidth antiAliasFilterBandwidth,
        void (*dataReadyInterrupt)())
{
    pinMode(m_int1Pin, INPUT);
    if (dataReadyInterrupt != nullptr)
    {
        CHECK_BOOL_ERROR(setupDataReadyinterrupt());
        attachInterrupt(m_int1Pin, dataReadyInterrupt, RISING);
    }

    // Disable FSYNC
    pinMode(m_int2Pin, OUTPUT);
    digitalWrite(m_int2Pin, false);

    setAccelerometerRangeInG(accelerometerRange);
    setGyroscopeRangeInDegPerS(gyroscopeRange);
    CHECK_BOOL_ERROR(setAccelerometerRangeAndOdr(accelerometerRange, odr));
    CHECK_BOOL_ERROR(setGyroscopeRangeAndOdr(gyroscopeRange, odr));
    CHECK_BOOL_ERROR(setAntiAliasFilterBandwidth(antiAliasFilterBandwidth));

    return setLowNoiseMode();
}

bool Icm40627::readData()
{
    constexpr uint8_t BLOCK_SIZE = 14;
    constexpr uint8_t BLOCK_ADDRESS = 0x1F;

    if (m_currentRegisterBank != RegisterBank::BANK_0)
    {
        CHECK_BOOL_ERROR(setRegisterBank(m_wire, m_address, RegisterBank::BANK_0));
        m_currentRegisterBank = RegisterBank::BANK_0;
    }

    m_wire.beginTransmission(m_address);
    m_wire.write(BLOCK_ADDRESS);
    if (m_wire.endTransmission(false) != 0)
    {
        return false;
    }

    if (m_wire.requestFrom(m_address, BLOCK_SIZE) != BLOCK_SIZE)
    {
        return false;
    }

    m_accelerationX = getInt16(m_wire.read(), m_wire.read());
    m_accelerationY = getInt16(m_wire.read(), m_wire.read());
    m_accelerationZ = getInt16(m_wire.read(), m_wire.read());
    m_angularRateX = getInt16(m_wire.read(), m_wire.read());
    m_angularRateY = getInt16(m_wire.read(), m_wire.read());
    m_angularRateZ = getInt16(m_wire.read(), m_wire.read());

    return true;
}

void Icm40627::setAccelerometerRangeInG(AccelerometerRange accelerometerRange)
{
    switch (accelerometerRange)
    {
    case AccelerometerRange::RANGE_2G:
        m_accelerometerRangeInG = 2.f;
        break;
    case AccelerometerRange::RANGE_4G:
        m_accelerometerRangeInG = 4.f;
        break;
    case AccelerometerRange::RANGE_8G:
        m_accelerometerRangeInG = 8.f;
        break;
    case AccelerometerRange::RANGE_16G:
        m_accelerometerRangeInG = 16.f;
        break;
    }
}

void Icm40627::setGyroscopeRangeInDegPerS(GyroscopeRange gyroscopeRange)
{
    switch (gyroscopeRange)
    {
    case GyroscopeRange::RANGE_15_625_DPS:
        m_gyroscopeRangeInDegPerS = 15.625f;
        break;
    case GyroscopeRange::RANGE_31_25_DPS:
        m_gyroscopeRangeInDegPerS = 31.25f;
        break;
    case GyroscopeRange::RANGE_62_5_DPS:
        m_gyroscopeRangeInDegPerS = 62.5f;
        break;
    case GyroscopeRange::RANGE_125_DPS:
        m_gyroscopeRangeInDegPerS = 125.f;
        break;
    case GyroscopeRange::RANGE_250_DPS:
        m_gyroscopeRangeInDegPerS = 250.f;
        break;
    case GyroscopeRange::RANGE_500_DPS:
        m_gyroscopeRangeInDegPerS = 500.f;
        break;
    case GyroscopeRange::RANGE_1000_DPS:
        m_gyroscopeRangeInDegPerS = 1000.f;
        break;
    case GyroscopeRange::RANGE_2000_DPS:
        m_gyroscopeRangeInDegPerS = 2000.f;
        break;
    }
}

bool Icm40627::setupDataReadyinterrupt()
{
    constexpr uint8_t INT_CONFIG_ADDRESS = 0x14;
    constexpr uint8_t INT_CONFIG1_ADDRESS = 0x64;
    constexpr uint8_t INT_SOURCE0_ADDRESS = 0x65;

    // INT1 (pulsed, push pull, active high), INT2 (pulsed, open drain, active low)
    CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_0, INT_CONFIG_ADDRESS, 0b00000011));

    // Route UI data ready to INT1
    CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_0, INT_SOURCE0_ADDRESS, 0b00001000));

    // Pulse duration = 8µs, de-assert duration disabled, int reset
    return writeRegister(RegisterBank::BANK_0, INT_CONFIG1_ADDRESS, 0b01110000);
}

bool Icm40627::setAccelerometerRangeAndOdr(AccelerometerRange accelerometerRange, Odr odr)
{
    constexpr uint8_t ACCELEROMETER_CONFIG0_ADDRESS = 0x50;
    uint8_t config0 = (static_cast<uint8_t>(accelerometerRange) << 5) | static_cast<uint8_t>(odr);

    return writeRegister(RegisterBank::BANK_0, ACCELEROMETER_CONFIG0_ADDRESS, config0);
}

bool Icm40627::setGyroscopeRangeAndOdr(GyroscopeRange gyroscopeRange, Odr odr)
{
    constexpr uint8_t GYROSCOPE_CONFIG0_ADDRESS = 0x4F;
    uint8_t config0 = (static_cast<uint8_t>(gyroscopeRange) << 5) | static_cast<uint8_t>(odr);

    return writeRegister(RegisterBank::BANK_0, GYROSCOPE_CONFIG0_ADDRESS, config0);
}

bool Icm40627::setAntiAliasFilterBandwidth(AntiAliasFilterBandwidth antiAliasFilterBandwidth)
{
    constexpr uint8_t ACCELEROMETER_CONFIG_STATIC2_ADDRESS = 0x03;
    constexpr uint8_t ACCELEROMETER_CONFIG_STATIC3_ADDRESS = 0x04;
    constexpr uint8_t ACCELEROMETER_CONFIG_STATIC4_ADDRESS = 0x05;

    constexpr uint8_t GYROSCOPE_CONFIG_STATIC2_ADDRESS = 0x0B;
    constexpr uint8_t GYROSCOPE_CONFIG_STATIC3_ADDRESS = 0x0C;
    constexpr uint8_t GYROSCOPE_CONFIG_STATIC4_ADDRESS = 0x0D;
    constexpr uint8_t GYROSCOPE_CONFIG_STATIC5_ADDRESS = 0x0E;

    constexpr uint8_t DELT[static_cast<uint8_t>(Icm40627::AntiAliasFilterBandwidth::BANDWIDTH_ALL)]
    {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41 ,42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63
    };

    constexpr uint16_t DELT_SQR[static_cast<uint8_t>(Icm40627::AntiAliasFilterBandwidth::BANDWIDTH_ALL)]
    {
        1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 122, 144, 170, 196, 224, 256, 288, 324, 360, 400, 440, 488, 528, 576,
        624, 680, 736, 784, 848, 896, 960, 1024, 1088, 1152, 1232, 1296, 1396, 1440, 1536, 1600, 1696, 1760, 1856,
        1952, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2720, 2816, 2944, 3008, 3136, 3264, 3392, 3456, 3584, 3712,
        3840, 3968
    };

    constexpr uint8_t BITSHIFT[static_cast<uint8_t>(Icm40627::AntiAliasFilterBandwidth::BANDWIDTH_ALL)]
    {
        15, 13, 12, 11, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    };

    if (antiAliasFilterBandwidth == AntiAliasFilterBandwidth::BANDWIDTH_ALL)
    {
        // Disable the accelerometer anti-alias filter
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_2, ACCELEROMETER_CONFIG_STATIC2_ADDRESS, 0x00));

        // Disable the gyroscope anti-alias filter
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_1, GYROSCOPE_CONFIG_STATIC2_ADDRESS, 0x00));
    }
    else
    {
        uint8_t delt = DELT[static_cast<uint8_t>(antiAliasFilterBandwidth)];
        uint16_t deltSqr = DELT_SQR[static_cast<uint8_t>(antiAliasFilterBandwidth)];
        uint8_t bitshift = BITSHIFT[static_cast<uint8_t>(antiAliasFilterBandwidth)];

        uint8_t lowerByteDeltSqr = static_cast<uint8_t>(deltSqr & 0x00FF);
        uint8_t upperByteDeltSqr = static_cast<uint8_t>((deltSqr & 0xFF00) >> 8);
        uint8_t bitshiftUpperByteDeltSqr = ((bitshift & 0x0F) << 4) | (upperByteDeltSqr & 0x0F);

        // Setup the accelerometer anti-alias filter
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_2, GYROSCOPE_CONFIG_STATIC2_ADDRESS, (delt << 1) | 0b1));
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_2, ACCELEROMETER_CONFIG_STATIC3_ADDRESS, lowerByteDeltSqr));
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_2, ACCELEROMETER_CONFIG_STATIC4_ADDRESS, bitshiftUpperByteDeltSqr));

        // Setup the gyroscope anti-alias filter
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_1, GYROSCOPE_CONFIG_STATIC2_ADDRESS, 0b0000'0010)); // Enable the filter
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_1, GYROSCOPE_CONFIG_STATIC3_ADDRESS, delt));
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_1, GYROSCOPE_CONFIG_STATIC4_ADDRESS, lowerByteDeltSqr));
        CHECK_BOOL_ERROR(writeRegister(RegisterBank::BANK_1, GYROSCOPE_CONFIG_STATIC5_ADDRESS, bitshiftUpperByteDeltSqr));
    }
    return true;
}

bool Icm40627::setLowNoiseMode()
{
    constexpr uint8_t POWER_MODE_ADDRESS = 0x4E;
    constexpr uint8_t LOW_POWER_MODE = 0b0101111;
    return writeRegister(RegisterBank::BANK_0, POWER_MODE_ADDRESS, LOW_POWER_MODE);
}

bool Icm40627::writeRegister(RegisterBank registerBank, uint8_t registerAddress, uint8_t value)
{
    if (m_currentRegisterBank != registerBank)
    {
        CHECK_BOOL_ERROR(setRegisterBank(m_wire, m_address, registerBank));
        m_currentRegisterBank = registerBank;
    }
    return ::writeRegister(m_wire, m_address, registerAddress, value);
}

int16_t Icm40627::getInt16(uint8_t upperByte, uint8_t lowerByte)
{
    return (static_cast<int16_t>(upperByte) << 8) | static_cast<int16_t>(lowerByte);
}
