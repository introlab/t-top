#include <SerialCommunication.h>

#include <gtest/gtest.h>

#define EXPECT_BUFFER_EQ(buffer, bytes, size)                                                                          \
    for (size_t i = 0; i < (size); i++)                                                                                \
    {                                                                                                                  \
        EXPECT_EQ((buffer).dataToRead()[i], (bytes)[i]) << "i=" << i;                                                  \
    }

TEST(AcknowledgmentPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    AcknowledgmentPayload payload;
    payload.receivedMessageId = 0x0102;

    SerialCommunicationBuffer<1> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(AcknowledgmentPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    AcknowledgmentPayload payload;
    payload.receivedMessageId = 0x0102;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), AcknowledgmentPayload::PAYLOAD_SIZE);
    EXPECT_EQ(buffer.dataToRead()[0], 0x02);
    EXPECT_EQ(buffer.dataToRead()[1], 0x01);
}

TEST(AcknowledgmentPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(AcknowledgmentPayload::readFrom(buffer).has_value());
}

TEST(AcknowledgmentPayloadTests, readFrom_shouldReturnThePayload)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x01));
    buffer.write(static_cast<uint8_t>(0x02));

    auto payload = AcknowledgmentPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->receivedMessageId, 0x0201);
}


TEST(BaseStatusPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    BaseStatusPayload payload;

    SerialCommunicationBuffer<1> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(BaseStatusPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    BaseStatusPayload payload;
    payload.isPsuConnected = true;
    payload.hasChargerError = false;
    payload.isBatteryCharging = true;
    payload.hasBatteryError = false;
    payload.stateOfCharge = 1.f;
    payload.current = 2.f;
    payload.voltage = 3.f;
    payload.onboardTemperature = 4.f;
    payload.externalTemperature = 5.f;
    payload.frontLightSensor = 6.f;
    payload.backLightSensor = 7.f;
    payload.leftLightSensor = 9.f;
    payload.rightLightSensor = 10.f;
    payload.volume = 0x11;
    payload.maximumVolume = 0x22;

    constexpr uint8_t EXPECT_DATA[] = {
        0x01, 0x00, 0x01, 0x00,  // isPsuConnected, hasChargerError, isBatteryCharging, hasBatteryError

        0x00, 0x00, 0x80, 0x3f,  // stateOfCharge
        0x00, 0x00, 0x00, 0x40,  // current
        0x00, 0x00, 0x40, 0x40,  // voltage
        0x00, 0x00, 0x80, 0x40,  // onboardTemperature
        0x00, 0x00, 0xA0, 0x40,  // externalTemperature
        0x00, 0x00, 0xC0, 0x40,  // frontLightSensor
        0x00, 0x00, 0xE0, 0x40,  // backLightSensor
        0x00, 0x00, 0x10, 0x41,  // leftLightSensor
        0x00, 0x00, 0x20, 0x41,  // rightLightSensor

        0x11, 0x22  // volume, maximumVolume
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), BaseStatusPayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(BaseStatusPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(BaseStatusPayload::readFrom(buffer).has_value());
}

TEST(BaseStatusPayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {
        0x01, 0x00, 0x01, 0x00,  // isPsuConnected, hasChargerError, isBatteryCharging, hasBatteryError

        0x00, 0x00, 0x80, 0x3f,  // stateOfCharge
        0x00, 0x00, 0x00, 0x40,  // current
        0x00, 0x00, 0x40, 0x40,  // voltage
        0x00, 0x00, 0x80, 0x40,  // onboardTemperature
        0x00, 0x00, 0xA0, 0x40,  // externalTemperature
        0x00, 0x00, 0xC0, 0x40,  // frontLightSensor
        0x00, 0x00, 0xE0, 0x40,  // backLightSensor
        0x00, 0x00, 0x10, 0x41,  // leftLightSensor
        0x00, 0x00, 0x20, 0x41,  // rightLightSensor

        0x11, 0x22  // volume, maximumVolume
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = BaseStatusPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_TRUE(payload->isPsuConnected);
    EXPECT_FALSE(payload->hasChargerError);
    EXPECT_TRUE(payload->isBatteryCharging);
    EXPECT_FALSE(payload->hasBatteryError);
    EXPECT_EQ(payload->stateOfCharge, 1.f);
    EXPECT_EQ(payload->current, 2.f);
    EXPECT_EQ(payload->voltage, 3.f);
    EXPECT_EQ(payload->onboardTemperature, 4.f);
    EXPECT_EQ(payload->externalTemperature, 5.f);
    EXPECT_EQ(payload->frontLightSensor, 6.f);
    EXPECT_EQ(payload->backLightSensor, 7.f);
    EXPECT_EQ(payload->leftLightSensor, 9.f);
    EXPECT_EQ(payload->rightLightSensor, 10.f);
    EXPECT_EQ(payload->volume, 0x11);
    EXPECT_EQ(payload->maximumVolume, 0x22);
}


TEST(ButtonPressedPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    ButtonPressedPayload payload;
    payload.button = Button::START;

    SerialCommunicationBuffer<0> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(ButtonPressedPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    ButtonPressedPayload payload;
    payload.button = Button::STOP;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), ButtonPressedPayload::PAYLOAD_SIZE);
    EXPECT_EQ(buffer.dataToRead()[0], 0x01);
}

TEST(ButtonPressedPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<0> buffer;
    EXPECT_FALSE(ButtonPressedPayload::readFrom(buffer).has_value());
}

TEST(ButtonPressedPayloadTests, readFrom_shouldReturnThePayload)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x01));

    auto payload = ButtonPressedPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->button, Button::STOP);
}


TEST(SetVolumePayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    SetVolumePayload payload;
    payload.volume = 0x12;

    SerialCommunicationBuffer<0> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(SetVolumePayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    SetVolumePayload payload;
    payload.volume = 0x56;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), SetVolumePayload::PAYLOAD_SIZE);
    EXPECT_EQ(buffer.dataToRead()[0], 0x56);
}

TEST(SetVolumePayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<0> buffer;
    EXPECT_FALSE(SetVolumePayload::readFrom(buffer).has_value());
}

TEST(SetVolumePayloadTests, readFrom_shouldReturnThePayload)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x89));

    auto payload = SetVolumePayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->volume, 0x89);
}


TEST(SetLedColorsPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    SetLedColorsPayload payload;

    SerialCommunicationBuffer<1> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(SetLedColorsPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    SetLedColorsPayload payload;
    for (size_t i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        payload.colors[i].red = i;
        payload.colors[i].green = i + 1;
        payload.colors[i].blue = i + 2;
    }

    constexpr uint8_t EXPECT_DATA[] = {0,  1,  2,  1,  2,  3,  2,  3,  4,  3,  4,  5,  4,  5,  6,  5,  6,  7,  6,
                                       7,  8,  7,  8,  9,  8,  9,  10, 9,  10, 11, 10, 11, 12, 11, 12, 13, 12, 13,
                                       14, 13, 14, 15, 14, 15, 16, 15, 16, 17, 16, 17, 18, 17, 18, 19, 18, 19, 20,
                                       19, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23, 24, 23, 24, 25, 24, 25, 26, 25,
                                       26, 27, 26, 27, 28, 27, 28, 29};

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), SetLedColorsPayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(SetLedColorsPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(SetLedColorsPayload::readFrom(buffer).has_value());
}

TEST(SetLedColorsPayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {1,  2,  3,  2,  3,  4,  3,  4,  5,  4,  5,  6,  5,  6,  7,  6,  7,  8,  7,
                                8,  9,  8,  9,  10, 9,  10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14,
                                15, 14, 15, 16, 15, 16, 17, 16, 17, 18, 17, 18, 19, 18, 19, 20, 19, 20, 21,
                                20, 21, 22, 21, 22, 23, 22, 23, 24, 23, 24, 25, 24, 25, 26, 25, 26, 27, 26,
                                27, 28, 27, 28, 29, 28, 29, 30, 29, 30, 31, 30, 31, 32, 31, 32, 33};

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = SetLedColorsPayload::readFrom(buffer);
    for (size_t i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        EXPECT_EQ(payload->colors[i].red, i + 1);
        EXPECT_EQ(payload->colors[i].green, i + 2);
        EXPECT_EQ(payload->colors[i].blue, i + 3);
    }
}


TEST(MotorStatusPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    MotorStatusPayload payload;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(MotorStatusPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    MotorStatusPayload payload;
    payload.torsoOrientation = 1.f;
    payload.torsoServoSpeed = 2;
    payload.headServoAngle1 = 3.f;
    payload.headServoAngle2 = 4.f;
    payload.headServoAngle3 = 5.f;
    payload.headServoAngle4 = 6.f;
    payload.headServoAngle5 = 7.f;
    payload.headServoAngle6 = 8.f;
    payload.headServoSpeed1 = 9;
    payload.headServoSpeed2 = 10;
    payload.headServoSpeed3 = 11;
    payload.headServoSpeed4 = 12;
    payload.headServoSpeed5 = 13;
    payload.headServoSpeed6 = 14;
    payload.headPosePositionX = 15.f;
    payload.headPosePositionY = 16.f;
    payload.headPosePositionZ = 17.f;
    payload.headPoseOrientationW = 18.f;
    payload.headPoseOrientationX = 19.f;
    payload.headPoseOrientationY = 20.f;
    payload.headPoseOrientationZ = 21.f;
    payload.isHeadPoseReachable = true;

    constexpr uint8_t EXPECT_DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // torsoOrientation
        0x02, 0x00,  // torsoServoSpeed
        0x00, 0x00, 0x40, 0x40,  // headServoAngle1
        0x00, 0x00, 0x80, 0x40,  // headServoAngle2
        0x00, 0x00, 0xA0, 0x40,  // headServoAngle3
        0x00, 0x00, 0xC0, 0x40,  // headServoAngle4
        0x00, 0x00, 0xE0, 0x40,  // headServoAngle5
        0x00, 0x00, 0x00, 0x41,  // headServoAngle6
        0x09, 0x00, 0x0A, 0x00,  // headServoSpeed1, headServoSpeed2
        0x0B, 0x00, 0x0C, 0x00,  // headServoSpeed3, headServoSpeed4
        0x0D, 0x00, 0x0E, 0x00,  // headServoSpeed5, headServoSpeed6
        0x00, 0x00, 0x70, 0x41,  // headPosePositionX
        0x00, 0x00, 0x80, 0x41,  // headPosePositionY
        0x00, 0x00, 0x88, 0x41,  // headPosePositionZ
        0x00, 0x00, 0x90, 0x41,  // headPoseOrientationW
        0x00, 0x00, 0x98, 0x41,  // headPoseOrientationX
        0x00, 0x00, 0xA0, 0x41,  // headPoseOrientationY
        0x00, 0x00, 0xA8, 0x41,  // headPoseOrientationZ
        0x01  // isHeadPoseReachable
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), MotorStatusPayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(MotorStatusPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(MotorStatusPayload::readFrom(buffer).has_value());
}

TEST(MotorStatusPayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // torsoOrientation
        0x02, 0x00,  // torsoServoSpeed
        0x00, 0x00, 0x40, 0x40,  // headServoAngle1
        0x00, 0x00, 0x80, 0x40,  // headServoAngle2
        0x00, 0x00, 0xA0, 0x40,  // headServoAngle3
        0x00, 0x00, 0xC0, 0x40,  // headServoAngle4
        0x00, 0x00, 0xE0, 0x40,  // headServoAngle5
        0x00, 0x00, 0x00, 0x41,  // headServoAngle6
        0x09, 0x00, 0x0A, 0x00,  // headServoSpeed1, headServoSpeed2
        0x0B, 0x00, 0x0C, 0x00,  // headServoSpeed3, headServoSpeed4
        0x0D, 0x00, 0x0E, 0x00,  // headServoSpeed5, headServoSpeed6
        0x00, 0x00, 0x70, 0x41,  // headPosePositionX
        0x00, 0x00, 0x80, 0x41,  // headPosePositionY
        0x00, 0x00, 0x88, 0x41,  // headPosePositionZ
        0x00, 0x00, 0x90, 0x41,  // headPoseOrientationW
        0x00, 0x00, 0x98, 0x41,  // headPoseOrientationX
        0x00, 0x00, 0xA0, 0x41,  // headPoseOrientationY
        0x00, 0x00, 0xA8, 0x41,  // headPoseOrientationZ
        0x01  // isHeadPoseReachable
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = MotorStatusPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->torsoOrientation, 1.f);
    EXPECT_EQ(payload->torsoServoSpeed, 2);
    EXPECT_EQ(payload->headServoAngle1, 3.f);
    EXPECT_EQ(payload->headServoAngle2, 4.f);
    EXPECT_EQ(payload->headServoAngle3, 5.f);
    EXPECT_EQ(payload->headServoAngle4, 6.f);
    EXPECT_EQ(payload->headServoAngle5, 7.f);
    EXPECT_EQ(payload->headServoAngle6, 8.f);
    EXPECT_EQ(payload->headServoSpeed1, 9);
    EXPECT_EQ(payload->headServoSpeed2, 10);
    EXPECT_EQ(payload->headServoSpeed3, 11);
    EXPECT_EQ(payload->headServoSpeed4, 12);
    EXPECT_EQ(payload->headServoSpeed5, 13);
    EXPECT_EQ(payload->headServoSpeed6, 14);
    EXPECT_EQ(payload->headPosePositionX, 15.f);
    EXPECT_EQ(payload->headPosePositionY, 16.f);
    EXPECT_EQ(payload->headPosePositionZ, 17.f);
    EXPECT_EQ(payload->headPoseOrientationW, 18.f);
    EXPECT_EQ(payload->headPoseOrientationX, 19.f);
    EXPECT_EQ(payload->headPoseOrientationY, 20.f);
    EXPECT_EQ(payload->headPoseOrientationZ, 21.f);
    EXPECT_EQ(payload->isHeadPoseReachable, true);
}


TEST(ImuDataPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    ImuDataPayload payload;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(ImuDataPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    ImuDataPayload payload;
    payload.accelerationX = 1.f;
    payload.accelerationY = 2.f;
    payload.accelerationZ = 3.f;
    payload.angularRateX = 4.f;
    payload.angularRateY = 5.f;
    payload.angularRateZ = 6.f;

    constexpr uint8_t EXPECT_DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // accelerationX
        0x00, 0x00, 0x00, 0x40,  // accelerationY
        0x00, 0x00, 0x40, 0x40,  // accelerationZ
        0x00, 0x00, 0x80, 0x40,  // angularRateX
        0x00, 0x00, 0xA0, 0x40,  // angularRateY
        0x00, 0x00, 0xC0, 0x40,  // angularRateZ
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), ImuDataPayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(ImuDataPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(ImuDataPayload::readFrom(buffer).has_value());
}

TEST(ImuDataPayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // accelerationX
        0x00, 0x00, 0x00, 0x40,  // accelerationY
        0x00, 0x00, 0x40, 0x40,  // accelerationZ
        0x00, 0x00, 0x80, 0x40,  // angularRateX
        0x00, 0x00, 0xA0, 0x40,  // angularRateY
        0x00, 0x00, 0xC0, 0x40,  // angularRateZ
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = ImuDataPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->accelerationX, 1.f);
    EXPECT_EQ(payload->accelerationY, 2);
    EXPECT_EQ(payload->accelerationZ, 3.f);
    EXPECT_EQ(payload->angularRateX, 4.f);
    EXPECT_EQ(payload->angularRateY, 5.f);
    EXPECT_EQ(payload->angularRateZ, 6.f);
}


TEST(SetTorsoOrientationPayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    SetTorsoOrientationPayload payload;

    SerialCommunicationBuffer<3> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(SetTorsoOrientationPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    SetTorsoOrientationPayload payload;
    payload.torsoOrientation = 1.f;

    constexpr uint8_t EXPECT_DATA[] = {
        0x00,
        0x00,
        0x80,
        0x3F,  // torsoOrientation
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), SetTorsoOrientationPayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(SetTorsoOrientationPayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(SetTorsoOrientationPayload::readFrom(buffer).has_value());
}

TEST(SetTorsoOrientationPayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {
        0x00,
        0x00,
        0x80,
        0x3F,  // torsoOrientation
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = SetTorsoOrientationPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->torsoOrientation, 1.f);
}


TEST(SetHeadPosePayloadTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    SetHeadPosePayload payload;

    SerialCommunicationBuffer<10> buffer;
    EXPECT_FALSE(payload.writeTo(buffer));
}

TEST(SetHeadPosePayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    SetHeadPosePayload payload;
    payload.headPosePositionX = 1.f;
    payload.headPosePositionY = 2.f;
    payload.headPosePositionZ = 3.f;
    payload.headPoseOrientationW = 4.f;
    payload.headPoseOrientationX = 5.f;
    payload.headPoseOrientationY = 6.f;
    payload.headPoseOrientationZ = 7.f;

    constexpr uint8_t EXPECT_DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // headPosePositionX
        0x00, 0x00, 0x00, 0x40,  // headPosePositionY
        0x00, 0x00, 0x40, 0x40,  // headPosePositionZ
        0x00, 0x00, 0x80, 0x40,  // headPoseOrientationW
        0x00, 0x00, 0xA0, 0x40,  // headPoseOrientationX
        0x00, 0x00, 0xC0, 0x40,  // headPoseOrientationY
        0x00, 0x00, 0xE0, 0x40,  // headPoseOrientationZ
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), SetHeadPosePayload::PAYLOAD_SIZE);
    EXPECT_BUFFER_EQ(buffer, EXPECT_DATA, sizeof(EXPECT_DATA));
}

TEST(SetHeadPosePayloadTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<100> buffer;
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(SetHeadPosePayload::readFrom(buffer).has_value());
}

TEST(SetHeadPosePayloadTests, readFrom_shouldReturnThePayload)
{
    constexpr uint8_t DATA[] = {
        0x00, 0x00, 0x80, 0x3F,  // headPosePositionX
        0x00, 0x00, 0x00, 0x40,  // headPosePositionY
        0x00, 0x00, 0x40, 0x40,  // headPosePositionZ
        0x00, 0x00, 0x80, 0x40,  // headPoseOrientationW
        0x00, 0x00, 0xA0, 0x40,  // headPoseOrientationX
        0x00, 0x00, 0xC0, 0x40,  // headPoseOrientationY
        0x00, 0x00, 0xE0, 0x40,  // headPoseOrientationZ
    };

    SerialCommunicationBuffer<100> buffer;
    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));

    auto payload = SetHeadPosePayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->headPosePositionX, 1.f);
    EXPECT_EQ(payload->headPosePositionY, 2);
    EXPECT_EQ(payload->headPosePositionZ, 3.f);
    EXPECT_EQ(payload->headPoseOrientationW, 4.f);
    EXPECT_EQ(payload->headPoseOrientationX, 5.f);
    EXPECT_EQ(payload->headPoseOrientationY, 6.f);
    EXPECT_EQ(payload->headPoseOrientationZ, 7.f);
}


TEST(ShutdownPayloadTests, writeTo_buffer_shouldWriteTheBytes)
{
    ShutdownPayload payload;

    SerialCommunicationBuffer<1> buffer;
    EXPECT_TRUE(payload.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), ShutdownPayload::PAYLOAD_SIZE);
}

TEST(ShutdownPayloadTests, readFrom_shouldReturnThePayload)
{
    SerialCommunicationBuffer<10> buffer;
    auto payload = ShutdownPayload::readFrom(buffer);
    ASSERT_TRUE(payload.has_value());
}
