#include <perception_logger/BinarySerialization.h>

#include <gtest/gtest.h>

#include <arpa/inet.h>

#include "BinarySerializationTests.h"

using namespace std;

TEST(BinarySerializationTests, isLittleEndian_shouldReturnTheRightValue)
{
    constexpr uint32_t VALUE = 157;
    EXPECT_EQ(isLittleEndian(), (htonl(VALUE) != VALUE));
}

TEST(BinarySerializationTests, switchEndianness_uint32_shouldSwitchEndianness)
{
    EXPECT_EQ(switchEndianness(static_cast<uint32_t>(0x01020304)), 0x04030201);
}

TEST(BinarySerializationTests, switchEndianness_float_shouldSwitchEndianness)
{
    EXPECT_FLOAT_EQ(switchEndianness(1.f), 4.6006e-41);
    EXPECT_FLOAT_EQ(switchEndianness(2.f), 8.9683102e-44);
    EXPECT_FLOAT_EQ(switchEndianness(3.f), 2.3048557e-41);
}

TEST(BinarySerializationTests, switchEndianness_uint64_shouldSwitchEndianness)
{
    EXPECT_EQ(switchEndianness(static_cast<uint64_t>(0x0102030405060708)), 0x0807060504030201);
}

TEST(BinarySerializationTests, switchEndianness_double_shouldSwitchEndianness)
{
    EXPECT_DOUBLE_EQ(switchEndianness(1.0), 3.0386519416174186e-319);
    EXPECT_DOUBLE_EQ(switchEndianness(2.0), 3.1620201333839779e-322);
    EXPECT_DOUBLE_EQ(switchEndianness(3.0), 1.0434666440167127e-320);
}

TEST(BinarySerializationTests, bytes_shouldBehaveLikeUniquePtr)
{
    int intValue = 56;
    Bytes notOwned(&intValue, sizeof(int));
    EXPECT_FALSE(notOwned.owned());
    EXPECT_EQ(notOwned.data(), &intValue);
    EXPECT_EQ(notOwned.size(), sizeof(int));

    Bytes owned(10);
    EXPECT_TRUE(owned.owned());
    EXPECT_NE(owned.data(), nullptr);
    EXPECT_EQ(owned.size(), 10);

    Bytes moved(move(notOwned));
    EXPECT_FALSE(moved.owned());
    EXPECT_EQ(moved.data(), &intValue);
    EXPECT_EQ(moved.size(), sizeof(int));
    EXPECT_FALSE(notOwned.owned());
    EXPECT_EQ(notOwned.data(), nullptr);
    EXPECT_EQ(notOwned.size(), 0);

    owned = move(moved);
    EXPECT_FALSE(owned.owned());
    EXPECT_EQ(owned.data(), &intValue);
    EXPECT_EQ(owned.size(), sizeof(int));
    EXPECT_FALSE(moved.owned());
    EXPECT_EQ(moved.data(), nullptr);
    EXPECT_EQ(moved.size(), 0);
}

TEST(BinarySerializationTests, serialize_uint32_shouldReturnLittleEndianBytes)
{
    constexpr uint32_t VALUE = 0x04030201;
    Bytes bytes = BinarySerializer<uint32_t>::serialize(VALUE);
    ASSERT_EQ(bytes.size(), 4);

    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(bytes.data());
    EXPECT_EQ(*ptr, nativeToLittleEndian(VALUE));
}

TEST(BinarySerializationTests, serialize_float_shouldReturnLittleEndianBytes)
{
    constexpr float VALUE = 7.f;
    Bytes bytes = BinarySerializer<float>::serialize(VALUE);
    ASSERT_EQ(bytes.size(), 4);

    const float* ptr = reinterpret_cast<const float*>(bytes.data());
    EXPECT_FLOAT_EQ(*ptr, nativeToLittleEndian(VALUE));
}

TEST(BinarySerializationTests, serialize_vectorFloat_shouldReturnLittleEndianBytes)
{
    constexpr float VALUE1 = 7.f;
    constexpr float VALUE2 = 8.f;
    vector<float> values = {VALUE1, VALUE2};
    Bytes bytes = BinarySerializer<vector<float>>::serialize(values);
    ASSERT_EQ(bytes.size(), 8);

    const float* ptr = reinterpret_cast<const float*>(bytes.data());
    EXPECT_EQ(ptr[0], nativeToLittleEndian(VALUE1));
    EXPECT_EQ(ptr[1], nativeToLittleEndian(VALUE2));
}
