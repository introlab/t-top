#ifndef DYNAMIXEL_CONTROL_SENSORS_ICM_40627_H
#define DYNAMIXEL_CONTROL_SENSORS_ICM_40627_H

#include <Arduino.h>
#include <Wire.h>

#include <limits>

class Icm40627
{
public:
    enum class AccelerometerRange : uint8_t
    {
        RANGE_2G = 0b011,
        RANGE_4G = 0b010,
        RANGE_8G = 0b001,
        RANGE_16G = 0b000
    };

    enum class GyroscopeRange : uint8_t
    {
        RANGE_15_625_DPS = 0b111,
        RANGE_31_25_DPS = 0b110,
        RANGE_62_5_DPS = 0b101,
        RANGE_125_DPS = 0b100,
        RANGE_250_DPS = 0b011,
        RANGE_500_DPS = 0b010,
        RANGE_1000_DPS = 0b001,
        RANGE_2000_DPS = 0b000
    };

    enum class Odr : uint8_t
    {
        ODR_12_5_HZ = 0b1011,
        ODR_25_HZ = 0b1010,
        ODR_50_HZ = 0b1001,
        ODR_100_HZ = 0b1000,
        ODR_200_HZ = 0b0111,
        ODR_500_HZ = 0b1111,
        ODR_1000_HZ = 0b0110,
        ODR_2000_HZ = 0b0101,
        ODR_4000_HZ = 0b0100,
        ODR_8000_HZ = 0b0011,
    };

    enum class AntiAliasFilterBandwidth : uint8_t
    {
        BANDWIDTH_10_HZ,
        BANDWIDTH_21_HZ,
        BANDWIDTH_32_HZ,
        BANDWIDTH_42_HZ,
        BANDWIDTH_53_HZ,
        BANDWIDTH_64_HZ,
        BANDWIDTH_76_HZ,
        BANDWIDTH_87_HZ,
        BANDWIDTH_99_HZ,
        BANDWIDTH_110_HZ,
        BANDWIDTH_122_HZ,
        BANDWIDTH_134_HZ,
        BANDWIDTH_146_HZ,
        BANDWIDTH_158_HZ,
        BANDWIDTH_171_HZ,
        BANDWIDTH_184_HZ,
        BANDWIDTH_196_HZ,
        BANDWIDTH_209_HZ,
        BANDWIDTH_222_HZ,
        BANDWIDTH_236_HZ,
        BANDWIDTH_249_HZ,
        BANDWIDTH_263_HZ,
        BANDWIDTH_277_HZ,
        BANDWIDTH_291_HZ,
        BANDWIDTH_305_HZ,
        BANDWIDTH_319_HZ,
        BANDWIDTH_334_HZ,
        BANDWIDTH_349_HZ,
        BANDWIDTH_364_HZ,
        BANDWIDTH_379_HZ,
        BANDWIDTH_394_HZ,
        BANDWIDTH_410_HZ,
        BANDWIDTH_425_HZ,
        BANDWIDTH_441_HZ,
        BANDWIDTH_458_HZ,
        BANDWIDTH_474_HZ,
        BANDWIDTH_490_HZ,
        BANDWIDTH_507_HZ,
        BANDWIDTH_524_HZ,
        BANDWIDTH_541_HZ,
        BANDWIDTH_559_HZ,
        BANDWIDTH_576_HZ,
        BANDWIDTH_594_HZ,
        BANDWIDTH_612_HZ,
        BANDWIDTH_631_HZ,
        BANDWIDTH_649_HZ,
        BANDWIDTH_668_HZ,
        BANDWIDTH_687_HZ,
        BANDWIDTH_706_HZ,
        BANDWIDTH_725_HZ,
        BANDWIDTH_745_HZ,
        BANDWIDTH_764_HZ,
        BANDWIDTH_784_HZ,
        BANDWIDTH_804_HZ,
        BANDWIDTH_825_HZ,
        BANDWIDTH_845_HZ,
        BANDWIDTH_866_HZ,
        BANDWIDTH_887_HZ,
        BANDWIDTH_908_HZ,
        BANDWIDTH_930_HZ,
        BANDWIDTH_951_HZ,
        BANDWIDTH_973_HZ,
        BANDWIDTH_995_HZ,
        BANDWIDTH_ALL
    };

    enum class RegisterBank : uint8_t
    {
        BANK_0 = 0b000,
        BANK_1 = 0b001,
        BANK_2 = 0b010,
        BANK_3 = 0b100
    };

private:
    TwoWire& m_wire;
    uint8_t m_int1Pin;
    uint8_t m_int2Pin;
    uint8_t m_address;
    RegisterBank m_currentRegisterBank;

    float m_accelerometerRangeInG;
    float m_gyroscopeRangeInDegPerS;

    int16_t m_accelerationX;
    int16_t m_accelerationY;
    int16_t m_accelerationZ;
    int16_t m_angularRateX;
    int16_t m_angularRateY;
    int16_t m_angularRateZ;

public:
    Icm40627(TwoWire& m_wire, uint8_t m_int1Pin, uint8_t m_int2Pin, uint8_t address);

    bool begin(AccelerometerRange accelerometerRange,
        GyroscopeRange gyroscopeRange,
        Odr odr,
        AntiAliasFilterBandwidth antiAliasFilterBandwidth);

    bool begin(AccelerometerRange accelerometerRange,
        GyroscopeRange gyroscopeRange,
        Odr odr,
        AntiAliasFilterBandwidth antiAliasFilterBandwidth,
        void (*dataReadyInterrupt)());

    bool readData();

    float getAccelerationXInMPerSS();
    float getAccelerationYInMPerSS();
    float getAccelerationZInMPerSS();

    float getAngularRateXInRadPerS();
    float getAngularRateYInRadPerS();
    float getAngularRateZInRadPerS();

private:
    void setAccelerometerRangeInG(AccelerometerRange accelerometerRange);
    void setGyroscopeRangeInDegPerS(GyroscopeRange gyroscopeRange);

    bool setupDataReadyinterrupt();
    bool setAccelerometerRangeAndOdr(AccelerometerRange accelerometerRange, Odr odr);
    bool setGyroscopeRangeAndOdr(GyroscopeRange gyroscopeRange, Odr odr);
    bool setAntiAliasFilterBandwidth(AntiAliasFilterBandwidth antiAliasFilterBandwidth);
    bool setLowNoiseMode();

    bool writeRegister(RegisterBank registerBank, uint8_t registerAddress, uint8_t value);

    int16_t getInt16(uint8_t upperByte, uint8_t lowerByte);

    float convertAccelerationToMPerSS(int16_t v);
    float convertAngularRateToRadPerS(int16_t v);
};

inline float Icm40627::getAccelerationXInMPerSS()
{
    return convertAccelerationToMPerSS(m_accelerationX);
}

inline float Icm40627::getAccelerationYInMPerSS()
{
    return convertAccelerationToMPerSS(m_accelerationY);
}

inline float Icm40627::getAccelerationZInMPerSS()
{
    return convertAccelerationToMPerSS(m_accelerationZ);
}

inline float Icm40627::getAngularRateXInRadPerS()
{
    return convertAngularRateToRadPerS(m_angularRateX);
}

inline float Icm40627::getAngularRateYInRadPerS()
{
    return convertAngularRateToRadPerS(m_angularRateY);
}

inline float Icm40627::getAngularRateZInRadPerS()
{
    return convertAngularRateToRadPerS(m_angularRateZ);
}

inline float Icm40627::convertAccelerationToMPerSS(int16_t v)
{
    return static_cast<float>(v) / static_cast<float>(std::numeric_limits<int16_t>::max()) * m_accelerometerRangeInG * 9.80665f;
}

inline float Icm40627::convertAngularRateToRadPerS(int16_t v)
{
    return static_cast<float>(v) / static_cast<float>(std::numeric_limits<int16_t>::max()) * m_gyroscopeRangeInDegPerS * 0.0174532925f;
}

#endif
