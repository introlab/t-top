#include <SerialCommunication.h>

#include <gtest/gtest.h>

enum class TestEnum : uint8_t
{
    A = 0
};

template<>
struct enum_min<TestEnum> : std::integral_constant<uint8_t, 0>
{
};

template<>
struct enum_max<TestEnum> : std::integral_constant<uint8_t, 2>
{
};


TEST(SerialCommunicationBufferTests, maxSize_shouldReturnN)
{
    SerialCommunicationBuffer<10> buffer;
    EXPECT_EQ(buffer.maxSize(), 10);
}

TEST(SerialCommunicationBufferTests, write_notEnoughSpace_shouldReturnFalse)
{
    SerialCommunicationBuffer<1> buffer;
    EXPECT_FALSE(buffer.write(static_cast<uint16_t>(0x0102)));
}

TEST(SerialCommunicationBufferTests, write_shouldSetData)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_TRUE(buffer.write(static_cast<uint16_t>(0x0102)));
    EXPECT_EQ(buffer.size(), 2);

    EXPECT_TRUE(buffer.write(static_cast<uint8_t>(0x03)));
    EXPECT_EQ(buffer.size(), 3);

    EXPECT_EQ(buffer[0], 0x02);
    EXPECT_EQ(buffer[1], 0x01);
    EXPECT_EQ(buffer[2], 0x03);
}

TEST(SerialCommunicationBufferTests, write_dataTooSmallBuffer_shouldReturnFalse)
{
    constexpr uint8_t DATA[] = {0x01, 0x02, 0x03, 0x04};
    SerialCommunicationBuffer<3> buffer;

    EXPECT_FALSE(buffer.write(DATA, sizeof(DATA)));
}

TEST(SerialCommunicationBufferTests, write_data_shouldCopyTheData)
{
    constexpr uint8_t DATA[] = {0x01, 0x02, 0x03, 0x04};
    SerialCommunicationBuffer<10> buffer;

    EXPECT_TRUE(buffer.write(DATA, sizeof(DATA)));
    EXPECT_EQ(buffer.size(), 4);

    EXPECT_EQ(buffer[0], 0x01);
    EXPECT_EQ(buffer[1], 0x02);
    EXPECT_EQ(buffer[2], 0x03);
    EXPECT_EQ(buffer[3], 0x04);
}

TEST(SerialCommunicationBufferTests, clear_shouldResetTheBuffer)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_TRUE(buffer.write(static_cast<uint16_t>(0x0102)));
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_EQ(buffer.read<uint16_t>(), 0x0102);

    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);
}

TEST(SerialCommunicationBufferTests, read_notEnoughSpace_shouldReturnNullopt)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_EQ(buffer.read<uint8_t>(), tl::nullopt);
    EXPECT_EQ(buffer.sizeToRead(), 0);

    EXPECT_TRUE(buffer.write(static_cast<uint8_t>(0x01)));
    EXPECT_EQ(buffer.sizeToRead(), 1);
    EXPECT_EQ(buffer.read<uint16_t>(), tl::nullopt);

    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);
}

TEST(SerialCommunicationBufferTests, read_shouldReadTheData)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_TRUE(buffer.write(static_cast<uint8_t>(0x01)));
    EXPECT_EQ(buffer.sizeToRead(), 1);
    EXPECT_TRUE(buffer.write(static_cast<uint16_t>(0x0203)));
    EXPECT_EQ(buffer.sizeToRead(), 3);

    EXPECT_EQ(buffer.read<uint8_t>(), 0x01);
    EXPECT_EQ(buffer.sizeToRead(), 2);

    EXPECT_EQ(buffer.read<uint16_t>(), 0x0203);
    EXPECT_EQ(buffer.sizeToRead(), 0);
    EXPECT_EQ(buffer.size(), 3);

    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.sizeToRead(), 0);
}

TEST(SerialCommunicationBufferTests, read_enum_shouldReadTheEnumValue)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_TRUE(buffer.write(static_cast<uint8_t>(0)));
    EXPECT_EQ(buffer.read<TestEnum>(), TestEnum::A);
}

TEST(SerialCommunicationBufferTests, read_invalidValueEnum_shouldReadTheEnumValue)
{
    SerialCommunicationBuffer<3> buffer;

    EXPECT_TRUE(buffer.write(static_cast<uint8_t>(10)));
    EXPECT_EQ(buffer.read<TestEnum>(), tl::nullopt);
}


TEST(SerialCommunicationBufferViewTests, maxSize_shouldReturnN)
{
    SerialCommunicationBuffer<10> buffer;
    SerialCommunicationBufferView view(buffer);
    EXPECT_EQ(view.maxSize(), 10);
}

TEST(SerialCommunicationBufferViewTests, write_notEnoughSpace_shouldReturnFalse)
{
    SerialCommunicationBuffer<1> buffer;
    SerialCommunicationBufferView view(buffer);
    EXPECT_FALSE(view.write(static_cast<uint16_t>(0x0102)));
}

TEST(SerialCommunicationBufferViewTests, write_shouldSetData)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint16_t>(0x0102)));
    EXPECT_EQ(view.size(), 2);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x03)));
    EXPECT_EQ(view.size(), 3);

    EXPECT_EQ(view[0], 0x02);
    EXPECT_EQ(view[1], 0x01);
    EXPECT_EQ(view[2], 0x03);
}

TEST(SerialCommunicationBufferViewTests, write_bool_shouldSet1Or0)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(true));
    EXPECT_TRUE(view.write(false));
    EXPECT_EQ(view.size(), 2);

    EXPECT_EQ(view[0], 0x01);
    EXPECT_EQ(view[1], 0x00);
}

TEST(SerialCommunicationBufferViewTests, write_dataTooSmallBuffer_shouldReturnFalse)
{
    constexpr uint8_t DATA[] = {0x01, 0x02, 0x03, 0x04};
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_FALSE(view.write(DATA, sizeof(DATA)));
}

TEST(SerialCommunicationBufferViewTests, write_data_shouldCopyTheData)
{
    constexpr uint8_t DATA[] = {0x01, 0x02, 0x03, 0x04};
    SerialCommunicationBuffer<10> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(DATA, sizeof(DATA)));
    EXPECT_EQ(view.size(), 4);

    EXPECT_EQ(view[0], 0x01);
    EXPECT_EQ(view[1], 0x02);
    EXPECT_EQ(view[2], 0x03);
    EXPECT_EQ(view[3], 0x04);
}

TEST(SerialCommunicationBufferViewTests, clear_shouldResetTheBuffer)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint16_t>(0x0102)));
    EXPECT_EQ(view.size(), 2);
    EXPECT_EQ(view.read<uint16_t>(), 0x0102);

    view.clear();
    EXPECT_EQ(view.size(), 0);
}

TEST(SerialCommunicationBufferViewTests, read_notEnoughSpace_shouldReturnNullopt)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_EQ(view.read<uint8_t>(), tl::nullopt);
    EXPECT_EQ(view.sizeToRead(), 0);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x01)));
    EXPECT_EQ(view.sizeToRead(), 1);
    EXPECT_EQ(view.read<uint16_t>(), tl::nullopt);

    view.clear();
    EXPECT_EQ(view.size(), 0);
}

TEST(SerialCommunicationBufferViewTests, read_shouldReadTheData)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x01)));
    EXPECT_EQ(buffer.sizeToRead(), 1);
    EXPECT_EQ(view.sizeToRead(), 1);
    EXPECT_TRUE(view.write(static_cast<uint16_t>(0x0203)));
    EXPECT_EQ(buffer.sizeToRead(), 3);
    EXPECT_EQ(view.sizeToRead(), 3);

    EXPECT_EQ(view.read<uint8_t>(), 0x01);
    EXPECT_EQ(buffer.sizeToRead(), 2);
    EXPECT_EQ(view.sizeToRead(), 2);

    EXPECT_EQ(view.read<uint16_t>(), 0x0203);
    EXPECT_EQ(view.sizeToRead(), 0);
    EXPECT_EQ(view.size(), 3);

    view.clear();
    EXPECT_EQ(view.size(), 0);
    EXPECT_EQ(view.sizeToRead(), 0);
}

TEST(SerialCommunicationBufferViewTests, read_bool_shouldReturnTrueIfDifferentFrom0)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x00)));
    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x01)));
    EXPECT_TRUE(view.write(static_cast<uint8_t>(0x02)));

    EXPECT_EQ(view.read<bool>(), false);
    EXPECT_EQ(view.read<bool>(), true);
    EXPECT_EQ(view.read<bool>(), true);
}

TEST(SerialCommunicationBufferViewTests, read_enum_shouldReadTheEnumValue)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(0)));
    EXPECT_EQ(view.read<TestEnum>(), TestEnum::A);
}

TEST(SerialCommunicationBufferViewTests, read_invalidValueEnum_shouldReadTheEnumValue)
{
    SerialCommunicationBuffer<3> buffer;
    SerialCommunicationBufferView view(buffer);

    EXPECT_TRUE(view.write(static_cast<uint8_t>(10)));
    EXPECT_EQ(view.read<TestEnum>(), tl::nullopt);
}