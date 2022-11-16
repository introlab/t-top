#include <SerialCommunication.h>

#include <gtest/gtest.h>

TEST(MessageHeaderTests, constructor_shouldSetAttributes)
{
    MessageHeader::resetMessageIdCounter();

    MessageHeader header1(Device::COMPUTER, Device::PSU_CONTROL, false, MessageType::BASE_STATUS);
    EXPECT_EQ(header1.source(), Device::COMPUTER);
    EXPECT_EQ(header1.destination(), Device::PSU_CONTROL);
    EXPECT_EQ(header1.acknowledgmentNeeded(), false);
    EXPECT_EQ(header1.messageId(), 0);
    EXPECT_EQ(header1.messageType(), MessageType::BASE_STATUS);

    MessageHeader header2(Device::DYNAMIXEL_CONTROL, Device::COMPUTER, true, MessageType::BUTTON_PRESSED);
    EXPECT_EQ(header2.source(), Device::DYNAMIXEL_CONTROL);
    EXPECT_EQ(header2.destination(), Device::COMPUTER);
    EXPECT_EQ(header2.acknowledgmentNeeded(), true);
    EXPECT_EQ(header2.messageId(), 1);
    EXPECT_EQ(header2.messageType(), MessageType::BUTTON_PRESSED);
}

TEST(MessageHeaderTests, writeTo_tooSmallBuffer_shouldReturnFalse)
{
    MessageHeader::resetMessageIdCounter();
    MessageHeader header(Device::COMPUTER, Device::PSU_CONTROL, true, MessageType::BASE_STATUS);

    SerialCommunicationBuffer<5> buffer;
    EXPECT_FALSE(header.writeTo(buffer));
}

TEST(MessageHeaderTests, writeTo_buffer_shouldWriteTheBytes)
{
    MessageHeader::setMessageIdCounter(0x0102);
    MessageHeader header(Device::COMPUTER, Device::PSU_CONTROL, true, MessageType::BASE_STATUS);

    SerialCommunicationBuffer<10> buffer;
    EXPECT_TRUE(header.writeTo(buffer));
    ASSERT_EQ(buffer.sizeToRead(), MessageHeader::HEADER_SIZE);
    EXPECT_EQ(buffer.dataToRead()[0], 0x02);
    EXPECT_EQ(buffer.dataToRead()[1], 0x00);
    EXPECT_EQ(buffer.dataToRead()[2], 0x01);
    EXPECT_EQ(buffer.dataToRead()[3], 0x02);
    EXPECT_EQ(buffer.dataToRead()[4], 0x01);
    EXPECT_EQ(buffer.dataToRead()[5], 0x01);
    EXPECT_EQ(buffer.dataToRead()[6], 0x00);
}

TEST(MessageHeaderTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x00));
    buffer.write(static_cast<uint8_t>(0x00));
    buffer.write(static_cast<uint8_t>(0x00));
    buffer.write(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(MessageHeader::readFrom(buffer).has_value());
}

TEST(MessageHeaderTests, readFrom_shouldReturnTheHeader)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.write(static_cast<uint8_t>(0x01));
    buffer.write(static_cast<uint8_t>(0x02));
    buffer.write(static_cast<uint8_t>(0x00));
    buffer.write(static_cast<uint8_t>(0xFF));
    buffer.write(static_cast<uint8_t>(0xAA));
    buffer.write(static_cast<uint8_t>(0x08));
    buffer.write(static_cast<uint8_t>(0x00));

    auto header = MessageHeader::readFrom(buffer);
    ASSERT_TRUE(header.has_value());
    EXPECT_EQ(header->source(), Device::DYNAMIXEL_CONTROL);
    EXPECT_EQ(header->destination(), Device::COMPUTER);
    EXPECT_EQ(header->acknowledgmentNeeded(), false);
    EXPECT_EQ(header->messageId(), 0xAAFF);
    EXPECT_EQ(header->messageType(), MessageType::SET_HEAD_POSE);
}


TEST(MessageTests, constructor_defaultAcknowledgment_shouldSetRightValues)
{
    MessageHeader::resetMessageIdCounter();

    Message<AcknowledgmentPayload> testee1(Device::PSU_CONTROL, Device::COMPUTER, AcknowledgmentPayload{});
    EXPECT_EQ(testee1.header().source(), Device::PSU_CONTROL);
    EXPECT_EQ(testee1.header().destination(), Device::COMPUTER);
    EXPECT_FALSE(testee1.header().acknowledgmentNeeded());
    EXPECT_EQ(testee1.header().messageId(), 0);
    EXPECT_EQ(testee1.header().messageType(), MessageType::ACKNOWLEDGMENT);

    Message<BaseStatusPayload> testee2(Device::PSU_CONTROL, Device::COMPUTER, BaseStatusPayload{});
    EXPECT_FALSE(testee2.header().acknowledgmentNeeded());
    EXPECT_EQ(testee2.header().messageId(), 1);
    EXPECT_EQ(testee2.header().messageType(), MessageType::BASE_STATUS);

    Message<ButtonPressedPayload> testee3(Device::PSU_CONTROL, Device::COMPUTER, ButtonPressedPayload{});
    EXPECT_TRUE(testee3.header().acknowledgmentNeeded());
    EXPECT_EQ(testee3.header().messageType(), MessageType::BUTTON_PRESSED);

    Message<SetVolumePayload> testee4(Device::PSU_CONTROL, Device::COMPUTER, SetVolumePayload{});
    EXPECT_TRUE(testee4.header().acknowledgmentNeeded());
    EXPECT_EQ(testee4.header().messageType(), MessageType::SET_VOLUME);

    Message<SetLedColorsPayload> testee5(Device::PSU_CONTROL, Device::COMPUTER, SetLedColorsPayload{});
    EXPECT_TRUE(testee5.header().acknowledgmentNeeded());
    EXPECT_EQ(testee5.header().messageType(), MessageType::SET_LED_COLORS);

    Message<MotorStatusPayload> testee6(Device::PSU_CONTROL, Device::COMPUTER, MotorStatusPayload{});
    EXPECT_FALSE(testee6.header().acknowledgmentNeeded());
    EXPECT_EQ(testee6.header().messageType(), MessageType::MOTOR_STATUS);

    Message<ImuDataPayload> testee7(Device::PSU_CONTROL, Device::COMPUTER, ImuDataPayload{});
    EXPECT_FALSE(testee7.header().acknowledgmentNeeded());
    EXPECT_EQ(testee7.header().messageType(), MessageType::IMU_DATA);

    Message<SetTorsoOrientationPayload> testee8(Device::PSU_CONTROL, Device::COMPUTER, SetTorsoOrientationPayload{});
    EXPECT_TRUE(testee8.header().acknowledgmentNeeded());
    EXPECT_EQ(testee8.header().messageType(), MessageType::SET_TORSO_ORIENTATION);

    Message<SetHeadPosePayload> testee9(Device::PSU_CONTROL, Device::COMPUTER, SetHeadPosePayload{});
    EXPECT_TRUE(testee9.header().acknowledgmentNeeded());
    EXPECT_EQ(testee9.header().messageType(), MessageType::SET_HEAD_POSE);

    Message<ShutdownPayload> testee10(Device::PSU_CONTROL, Device::COMPUTER, ShutdownPayload{});
    EXPECT_TRUE(testee10.header().acknowledgmentNeeded());
    EXPECT_EQ(testee10.header().messageType(), MessageType::SHUTDOWN);
}

TEST(MessageTests, constructor_shouldSetRightValues)
{
    MessageHeader::resetMessageIdCounter();

    Message<ImuDataPayload> testee(Device::PSU_CONTROL, Device::COMPUTER, true, ImuDataPayload{});
    EXPECT_EQ(testee.header().source(), Device::PSU_CONTROL);
    EXPECT_EQ(testee.header().destination(), Device::COMPUTER);
    EXPECT_TRUE(testee.header().acknowledgmentNeeded());
    EXPECT_EQ(testee.header().messageId(), 0);
    EXPECT_EQ(testee.header().messageType(), MessageType::IMU_DATA);
}
