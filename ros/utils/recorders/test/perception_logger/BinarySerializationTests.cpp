#include <perception_logger/BinarySerialization.h>

#include <gtest/gtest.h>

#include <arpa/inet.h>

using namespace std;

TEST(BinarySerializationTests, isLittleEndian_shouldReturnTheRightValue)
{
    constexpr uint32_t VALUE = 157;
    EXPECT_EQ(isLittleEndian(), (htonl(VALUE) != VALUE));
}

TEST(BinarySerializationTests, toLittleEndianBytes_uint32_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 4> BYTES = {byte{4}, byte{3}, byte{2}, byte{1}};
    EXPECT_EQ(toLittleEndianBytes(static_cast<uint32_t>(0x01020304)), BYTES);
}

TEST(BinarySerializationTests, fromLittleEndianBytes_uint32_shouldSwitchEndiannessIfNeeded)
{
    constexpr array<byte, 4> BYTES = {byte{4}, byte{3}, byte{2}, byte{1}};
    EXPECT_EQ(fromLittleEndianBytes<uint32_t>(BYTES), static_cast<uint32_t>(0x01020304));
}

TEST(BinarySerializationTests, bytes_shouldBehaveLikeUniquePtr)
{
    int intValue = 56;
    Bytes notOwned(reinterpret_cast<byte*>(&intValue), sizeof(int));
    EXPECT_FALSE(notOwned.owned());
    EXPECT_EQ(notOwned.data(), reinterpret_cast<byte*>(&intValue));
    EXPECT_EQ(notOwned.size(), sizeof(int));

    Bytes owned(10);
    EXPECT_TRUE(owned.owned());
    EXPECT_NE(owned.data(), nullptr);
    EXPECT_EQ(owned.size(), 10);

    Bytes moved(move(notOwned));
    EXPECT_FALSE(moved.owned());
    EXPECT_EQ(moved.data(), reinterpret_cast<byte*>(&intValue));
    EXPECT_EQ(moved.size(), sizeof(int));
    EXPECT_FALSE(notOwned.owned());
    EXPECT_EQ(notOwned.data(), nullptr);
    EXPECT_EQ(notOwned.size(), 0);

    owned = move(moved);
    EXPECT_FALSE(owned.owned());
    EXPECT_EQ(owned.data(), reinterpret_cast<byte*>(&intValue));
    EXPECT_EQ(owned.size(), sizeof(int));
    EXPECT_FALSE(moved.owned());
    EXPECT_EQ(moved.data(), nullptr);
    EXPECT_EQ(moved.size(), 0);

    owned = move(owned);
    EXPECT_FALSE(owned.owned());
    EXPECT_EQ(owned.data(), reinterpret_cast<byte*>(&intValue));
    EXPECT_EQ(owned.size(), sizeof(int));
}

TEST(BinarySerializationTests, serializeToBytesNoCopy_uint32_shouldReturnLittleEndianBytes)
{
    constexpr uint32_t VALUE = 0x04030201;
    Bytes bytes = serializeToBytesNoCopy(VALUE);
    ASSERT_EQ(bytes.size(), 4);

    EXPECT_EQ(bytes.data()[0], byte{1});
    EXPECT_EQ(bytes.data()[1], byte{2});
    EXPECT_EQ(bytes.data()[2], byte{3});
    EXPECT_EQ(bytes.data()[3], byte{4});
}

TEST(BinarySerializationTests, serializeToBytesNoCopy_vectorUint32_shouldReturnLittleEndianBytes)
{
    vector<uint32_t> values = {0x04030201, 0x08070605};
    Bytes bytes = serializeToBytesNoCopy(values);
    ASSERT_EQ(bytes.size(), 8);

    const float* ptr = reinterpret_cast<const float*>(bytes.data());
    EXPECT_EQ(bytes.data()[0], byte{1});
    EXPECT_EQ(bytes.data()[1], byte{2});
    EXPECT_EQ(bytes.data()[2], byte{3});
    EXPECT_EQ(bytes.data()[3], byte{4});
    EXPECT_EQ(bytes.data()[4], byte{5});
    EXPECT_EQ(bytes.data()[5], byte{6});
    EXPECT_EQ(bytes.data()[6], byte{7});
    EXPECT_EQ(bytes.data()[7], byte{8});
}
