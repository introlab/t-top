#include <ego_noise_reduction/Math.h>

#include <gtest/gtest.h>

using namespace introlab;
using namespace std;

TEST(MathTests, hann_length0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(hann<arma::fvec>(0), NotSupportedException);
}

TEST(MathTests, hann_even_shouldReturnTheRightValues)
{
    constexpr size_t WindowLength = 4;
    arma::fvec window = hann<arma::fvec>(WindowLength);

    ASSERT_EQ(window.n_elem, WindowLength);
    EXPECT_FLOAT_EQ(window(0), 0.f);
    EXPECT_FLOAT_EQ(window(1), 0.75f);
    EXPECT_FLOAT_EQ(window(2), 0.75f);
    EXPECT_FLOAT_EQ(window(3), 0.f);
}

TEST(MathTests, hann_odd_shouldReturnTheRightValues)
{
    constexpr size_t WindowLength = 5;
    arma::fvec window = hann<arma::fvec>(WindowLength);

    ASSERT_EQ(window.n_elem, WindowLength);
    EXPECT_FLOAT_EQ(window(0), 0.f);
    EXPECT_FLOAT_EQ(window(1), 0.5f);
    EXPECT_FLOAT_EQ(window(2), 1.f);
    EXPECT_FLOAT_EQ(window(3), 0.5f);
    EXPECT_FLOAT_EQ(window(4), 0.f);
}
