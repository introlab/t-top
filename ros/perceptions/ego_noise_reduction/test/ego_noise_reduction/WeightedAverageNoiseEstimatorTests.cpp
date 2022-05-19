#include <ego_noise_reduction/WeightedAverageNoiseEstimator.h>
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

TEST(WeightedAverageNoiseEstimatorTests, constructor_invalidEpsilon_shouldThrowNotSupportedException)
{
    EXPECT_THROW(WeightedAverageNoiseEstimator(1, 2, -0.1f, 0.9f, 0.9f), NotSupportedException);
}

TEST(WeightedAverageNoiseEstimatorTests, constructor_invalidAlpha_shouldThrowNotSupportedException)
{
    EXPECT_THROW(WeightedAverageNoiseEstimator(1, 2, 0.1f, 1.1f, 0.9f), NotSupportedException);
    EXPECT_THROW(WeightedAverageNoiseEstimator(1, 2, 0.1f, -0.1f, 0.9f), NotSupportedException);
}

TEST(WeightedAverageNoiseEstimatorTests, constructor_invalidDelta_shouldThrowNotSupportedException)
{
    EXPECT_THROW(WeightedAverageNoiseEstimator(1, 2, 0.1f, 0.9f, 1.1f), NotSupportedException);
    EXPECT_THROW(WeightedAverageNoiseEstimator(1, 2, 0.1f, 0.9f, -0.1f), NotSupportedException);
}

TEST(WeightedAverageNoiseEstimatorTests, estimate_shouldEstimateTheNoise)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_wa_ne_output.raw", Format, ChannelCount, FrameSampleCount);

    auto testee = make_shared<WeightedAverageNoiseEstimator>(
        ChannelCount,
        FrameSampleCount,
        NoiseEstimatorEpsilon,
        NoiseEstimatorAlpha,
        NoiseEstimatorDelta);
    SpectralSubtractionNoiseRemover noiseRemover(
        ChannelCount,
        FrameSampleCount,
        testee,
        SpectralSubtractionAlpha0,
        SpectralSubtractionGamma,
        SpectralSubtractionBeta);
    testNoiseReduction(noiseRemover, inputPcmFrames, expectedOutputPcmFrames);
}
