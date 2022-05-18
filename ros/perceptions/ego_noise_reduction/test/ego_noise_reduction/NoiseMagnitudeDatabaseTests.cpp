#include <ego_noise_reduction/NoiseMagnitudeDatabase.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include "TestUtils.h"

#include <gtest/gtest.h>

using namespace introlab;
using namespace std;

constexpr size_t MagnitudeSize = 1025;
constexpr int MinSpeed = 10;
constexpr int MaxSpeed = 80;
constexpr int SpeedStep = 10;
constexpr int MinOrientation = 0;
constexpr int MaxOrientation = 350;
constexpr int OrientationStep = 10;
constexpr size_t ChannelCount = 16;

TEST(NoiseMagnitudeDatabaseTests, constructor_invalidSpeedStep_shouldThrowNotSupportedException)
{
    string directory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        NoiseMagnitudeDatabase(directory, "torso_servo", MagnitudeSize, 10, 80, 5, 0, 350, 10, 16),
        NotSupportedException);
}

TEST(NoiseMagnitudeDatabaseTests, constructor_invalidOrientationStep_shouldThrowNotSupportedException)
{
    string directory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        NoiseMagnitudeDatabase(directory, "torso_servo", MagnitudeSize, 10, 80, 10, 0, 350, 5, 16),
        NotSupportedException);
}

TEST(NoiseMagnitudeDatabaseTests, constructor_invalidChannelCount_shouldThrowNotSupportedException)
{
    string directory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        NoiseMagnitudeDatabase(directory, "torso_servo", MagnitudeSize, 10, 80, 10, 0, 350, 10, 17),
        NotSupportedException);
}

TEST(NoiseMagnitudeDatabaseTests, constructor_invalidMagnitudeSize_shouldThrowNotSupportedException)
{
    string directory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        NoiseMagnitudeDatabase(directory, "torso_servo", MagnitudeSize - 1, 10, 80, 10, 0, 350, 10, 17),
        NotSupportedException);
}

TEST(NoiseMagnitudeDatabaseTests, noiseMagnitude_shouldSetTheNoiseMagnitude)
{
    constexpr int Speed = 80;
    constexpr int Orinetation = 0;
    constexpr size_t ChannelIndex = 2;
    NoiseMagnitudeDatabase testee(
        getResourcesPath() + "/noise_data",
        "torso_servo",
        MagnitudeSize,
        MinSpeed,
        MaxSpeed,
        SpeedStep,
        MinOrientation,
        MaxOrientation,
        OrientationStep,
        ChannelCount);

    arma::fvec noiseMagnitude;
    testee.noiseMagnitude(noiseMagnitude, Speed, Orinetation, ChannelIndex);

    ASSERT_EQ(noiseMagnitude.n_elem, 1025);
    EXPECT_FLOAT_EQ(noiseMagnitude[0], 0.2554924f);
    EXPECT_FLOAT_EQ(noiseMagnitude[1], 0.55282063f);
    EXPECT_FLOAT_EQ(noiseMagnitude[2], 0.99890353f);
    EXPECT_FLOAT_EQ(noiseMagnitude[3], 1.20199985f);
}
