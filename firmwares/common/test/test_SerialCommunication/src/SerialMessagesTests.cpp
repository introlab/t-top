#include <SerialCommunication.h>

#include <gtest/gtest.h>

TEST(MessageHeaderTests, constructor_shouldSetAttributes)
{
    MessageHeader::resetMessageIdCounter();

    MessageHeader header1(Device::COMPUTER, Device::PSU_CONTROL, false, MessageType::BASE_STATUS);
    EXPECT_EQ(header1.source(), Device::COMPUTER);
    EXPECT_EQ(header1.destimation(), Device::PSU_CONTROL);
    EXPECT_EQ(header1.acknowledgmentNeeded(), false);
    EXPECT_EQ(header1.messageId(), 0);
    EXPECT_EQ(header1.messageType(), MessageType::BASE_STATUS);

    MessageHeader header2(Device::DYNAMIXEL_CONTROL, Device::COMPUTER, true, MessageType::BUTTON_PRESSED);
    EXPECT_EQ(header2.source(), Device::DYNAMIXEL_CONTROL);
    EXPECT_EQ(header2.destimation(), Device::COMPUTER);
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
    ASSERT_EQ(buffer.size(), 7);
    EXPECT_EQ(buffer[0], 0x02);
    EXPECT_EQ(buffer[1], 0x00);
    EXPECT_EQ(buffer[2], 0x01);
    EXPECT_EQ(buffer[3], 0x02);
    EXPECT_EQ(buffer[4], 0x01);
    EXPECT_EQ(buffer[5], 0x01);
    EXPECT_EQ(buffer[6], 0x00);
}

TEST(MessageHeaderTests, readFrom_tooSmallBuffer_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.put(static_cast<uint8_t>(0x00));
    buffer.put(static_cast<uint8_t>(0x00));
    buffer.put(static_cast<uint8_t>(0x00));
    buffer.put(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(MessageHeader::readFrom(buffer).has_value());
}

TEST(MessageHeaderTests, readFrom_invalidValue_shouldReturnNullOpt)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.put(static_cast<uint8_t>(0x01));
    buffer.put(static_cast<uint8_t>(0x02));
    buffer.put(static_cast<uint8_t>(0x00));
    buffer.put(static_cast<uint8_t>(0xFF));
    buffer.put(static_cast<uint8_t>(0xFF));
    buffer.put(static_cast<uint8_t>(0xFF));
    buffer.put(static_cast<uint8_t>(0x00));

    EXPECT_FALSE(MessageHeader::readFrom(buffer).has_value());
}

TEST(MessageHeaderTests, readFrom_shouldReturnReturnTheHeader)
{
    SerialCommunicationBuffer<10> buffer;
    buffer.put(static_cast<uint8_t>(0x01));
    buffer.put(static_cast<uint8_t>(0x02));
    buffer.put(static_cast<uint8_t>(0x00));
    buffer.put(static_cast<uint8_t>(0xFF));
    buffer.put(static_cast<uint8_t>(0xAA));
    buffer.put(static_cast<uint8_t>(0x08));
    buffer.put(static_cast<uint8_t>(0x00));

    auto header = MessageHeader::readFrom(buffer);
    ASSERT_TRUE(header.has_value());
    EXPECT_EQ(header->source(), Device::DYNAMIXEL_CONTROL);
    EXPECT_EQ(header->destimation(), Device::COMPUTER);
    EXPECT_EQ(header->acknowledgmentNeeded(), false);
    EXPECT_EQ(header->messageId(), 0xAAFF);
    EXPECT_EQ(header->messageType(), MessageType::SET_HEAD_POSE);
}
