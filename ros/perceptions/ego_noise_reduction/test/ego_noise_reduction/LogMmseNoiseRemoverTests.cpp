#include <ego_noise_reduction/LogMmseNoiseRemover.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include "TestUtils.h"

#include <gtest/gtest.h>

using namespace introlab;
using namespace std;

constexpr float Alpha = 0.98f;
constexpr float MaxAPosterioriSnr = 40.f;
constexpr float MinAPrioriSnr = 0.003f;

constexpr float AbsError = 0.001f;

TEST(LogMmseNoiseRemoverTests, constructor_negativeAlpha_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        LogMmseNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), -0.1f, 0.1f, 0.01f),
        NotSupportedException);
}

TEST(LogMmseNoiseRemoverTests, constructor_maxAPosterioriSnr_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        LogMmseNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), 4.f, -0.1f, 0.01f),
        NotSupportedException);
}

TEST(LogMmseNoiseRemoverTests, constructor_minAPrioriSnr_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        LogMmseNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), 4.f, 0.1f, -0.01f),
        NotSupportedException);
}

TEST(LogMmseNoiseRemoverTests, removeNoise_zeroNoiseMagnitudeSpectrum_shouldReturnTheSameSignal)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(
        resourcesPath + "/noisy_sounds_log_mmse_zero_output.raw",
        Format,
        ChannelCount,
        FrameSampleCount);

    LogMmseNoiseRemover testee(
        ChannelCount,
        FrameSampleCount,
        createZeroConstantNoiseEstimator(ChannelCount, FrameSampleCount),
        Alpha,
        MaxAPosterioriSnr,
        MinAPrioriSnr);
    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames);
}

TEST(LogMmseNoiseRemoverTests, removeNoise_shouldRemoveTheNoise)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_log_mmse_output.raw", Format, ChannelCount, FrameSampleCount);

    LogMmseNoiseRemover testee(
        ChannelCount,
        FrameSampleCount,
        createConstantNoiseEstimatorFromFile(resourcesPath + "/noises.txt"),
        Alpha,
        MaxAPosterioriSnr,
        MinAPrioriSnr);
    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames);
}
