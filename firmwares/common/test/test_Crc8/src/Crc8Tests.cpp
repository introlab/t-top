#include <Crc8.h>

#include <gtest/gtest.h>

TEST(Crc8Tests, crc8_shouldGetTheRightResult)
{
    constexpr uint8_t DATA[] = {0x20, 0x3D, 0x03, 0x01, 0x0C, 0x62};
    EXPECT_EQ(crc8(DATA, sizeof(DATA)), 0x1D);
}
