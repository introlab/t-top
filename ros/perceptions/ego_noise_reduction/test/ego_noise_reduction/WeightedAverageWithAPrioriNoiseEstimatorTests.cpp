#include <ego_noise_reduction/WeightedAverageWithAPrioriNoiseEstimator.h>
#include <ego_noise_reduction/SpectralSubtractionNoiseRemover.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include "TestUtils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace introlab;
using namespace std;

constexpr float NoiseEstimatorEpsilon = 4.f;
constexpr float NoiseEstimatorAlpha = 0.9f;
constexpr float NoiseEstimatorDelta = 0.9f;

constexpr float SpectralSubtractionAlpha0 = 4.f;
constexpr float SpectralSubtractionGamma = 0.1f;
constexpr float SpectralSubtractionBeta = 0.01f;

constexpr float AbsError = 0.001f;

TEST(WeightedAverageWithAPrioriNoiseEstimatorTests, round_shouldReturnRightValues)
{
    EXPECT_EQ(round(0.f, 1.f, 5.f), 1);
    EXPECT_EQ(round(2.f, 1.f, 5.f), 1);
    EXPECT_EQ(round(3.f, 1.f, 5.f), 1);
    EXPECT_EQ(round(3.5f, 1.f, 5.f), 6);

    EXPECT_EQ(round(3.f, 20.f, 10.f), 0);
    EXPECT_EQ(round(5.1f, 20.f, 10.f), 10);
    EXPECT_EQ(round(11.f, 20.f, 10.f), 10);

    EXPECT_EQ(round(24.f, 20.f, 10.f), 20);
    EXPECT_EQ(round(37.f, 20.f, 10.f), 40);
}

TEST(WeightedAverageWithAPrioriNoiseEstimatorTests, convertSpeed_shouldReturnRightValues)
{
    EXPECT_EQ(convertSpeed(0, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(5, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(19, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(20, MaxTorsoSpeed), 20);
    EXPECT_EQ(convertSpeed(-1, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(-8, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(-19, MaxTorsoSpeed), 0);
    EXPECT_EQ(convertSpeed(-20, MaxTorsoSpeed), 20);

    EXPECT_EQ(convertSpeed(23, MaxTorsoSpeed), 25);
    EXPECT_EQ(convertSpeed(90, MaxTorsoSpeed), 80);
}

TEST(WeightedAverageWithAPrioriNoiseEstimatorTests, convertOrientationRadianToDegrees_shouldReturnRightValues)
{
    EXPECT_EQ(convertOrientationRadianToDegrees(0.174533f), 10);
    EXPECT_EQ(convertOrientationRadianToDegrees(-0.0873f), 350);
    EXPECT_EQ(convertOrientationRadianToDegrees(2 * M_PI), 00);
    EXPECT_EQ(convertOrientationRadianToDegrees(2 * M_PI + 0.174533f), 10);
    EXPECT_EQ(convertOrientationRadianToDegrees(2 * M_PI - 0.279253f), 340);
}

TEST(WeightedAverageWithAPrioriNoiseEstimatorTests, constructor_invalidChannelCount_shouldThrowNotSupportedException)
{
    string noiseDirectory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        WeightedAverageWithAPrioriNoiseEstimator(
            15,
            2048,
            44100,
            NoiseEstimatorEpsilon,
            NoiseEstimatorAlpha,
            NoiseEstimatorDelta,
            noiseDirectory),
        NotSupportedException);
}

TEST(
    WeightedAverageWithAPrioriNoiseEstimatorTests,
    constructor_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    string noiseDirectory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        WeightedAverageWithAPrioriNoiseEstimator(
            16,
            2047,
            44100,
            NoiseEstimatorEpsilon,
            NoiseEstimatorAlpha,
            NoiseEstimatorDelta,
            noiseDirectory),
        NotSupportedException);
}

TEST(
    WeightedAverageWithAPrioriNoiseEstimatorTests,
    constructor_invalidSamplingFrequency_shouldThrowNotSupportedException)
{
    string noiseDirectory = getResourcesPath() + "/noise_data";
    EXPECT_THROW(
        WeightedAverageWithAPrioriNoiseEstimator(
            16,
            2048,
            48000,
            NoiseEstimatorEpsilon,
            NoiseEstimatorAlpha,
            NoiseEstimatorDelta,
            noiseDirectory),
        NotSupportedException);
}

TEST(WeightedAverageWithAPrioriNoiseEstimatorTests, estimate_shouldEstimateTheNoise)
{
    constexpr size_t ChannelCount = 16;
    constexpr size_t FrameSampleCount = 2048;
    constexpr size_t SamplingFrequency = 44100;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_all_motors.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(
        resourcesPath + "/noisy_sounds_wa_ne_a_priori_output.raw",
        Format,
        ChannelCount,
        FrameSampleCount);

    auto testee = make_shared<WeightedAverageWithAPrioriNoiseEstimator>(
        ChannelCount,
        FrameSampleCount,
        SamplingFrequency,
        NoiseEstimatorEpsilon,
        NoiseEstimatorAlpha,
        NoiseEstimatorDelta,
        resourcesPath + "/noise_data");
    SpectralSubtractionNoiseRemover noiseRemover(
        ChannelCount,
        FrameSampleCount,
        testee,
        SpectralSubtractionAlpha0,
        SpectralSubtractionGamma,
        SpectralSubtractionBeta);

    testee->setTorsoSpeed(71);
    testee->setHeadSpeedId1(102);
    testee->setHeadSpeedId2(123);
    testee->setHeadSpeedId3(134);
    testee->setHeadSpeedId4(202);
    testee->setHeadSpeedId5(174);
    testee->setHeadSpeedId6(51);
    testee->setOrientationRadians(0.174533f);

    EXPECT_EQ(testee->torsoSpeed(), 70);
    EXPECT_EQ(testee->headSpeedId1(), 100);
    EXPECT_EQ(testee->headSpeedId2(), 125);
    EXPECT_EQ(testee->headSpeedId3(), 135);
    EXPECT_EQ(testee->headSpeedId4(), 200);
    EXPECT_EQ(testee->headSpeedId5(), 175);
    EXPECT_EQ(testee->headSpeedId6(), 50);
    EXPECT_EQ(testee->orientationDegrees(), 10);

    testNoiseReduction(noiseRemover, inputPcmFrames, expectedOutputPcmFrames);
}
