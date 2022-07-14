#include <ego_noise_reduction/SpectralSubtractionNoiseRemover.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include "TestUtils.h"

#include <gtest/gtest.h>

using namespace introlab;
using namespace std;

constexpr float Alpha0 = 4.f;
constexpr float Gamma = 0.1f;
constexpr float Beta = 0.01f;

constexpr float AbsError = 0.001f;

TEST(SpectralSubtractionNoiseRemoverTests, constructor_negativeAlpha0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        SpectralSubtractionNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), -0.1f, 0.1f, 0.01f),
        NotSupportedException);
}

TEST(SpectralSubtractionNoiseRemoverTests, constructor_negativeGamma_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        SpectralSubtractionNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), 4.f, -0.1f, 0.01f),
        NotSupportedException);
}

TEST(SpectralSubtractionNoiseRemoverTests, constructor_negativeBeta_shouldThrowNotSupportedException)
{
    EXPECT_THROW(
        SpectralSubtractionNoiseRemover(1, 4, createZeroConstantNoiseEstimator(1, 4), 4.f, 0.1f, -0.01f),
        NotSupportedException);
}

TEST(SpectralSubtractionNoiseRemoverTests, removeNoise_zeroNoiseMagnitudeSpectrum_shouldReturnTheSameSignal)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_zero_output.raw", Format, ChannelCount, FrameSampleCount);

    SpectralSubtractionNoiseRemover testee(
        ChannelCount,
        FrameSampleCount,
        createZeroConstantNoiseEstimator(ChannelCount, FrameSampleCount),
        Alpha0,
        Gamma,
        Beta);
    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames);
}

TEST(SpectralSubtractionNoiseRemoverTests, removeNoise_shouldRemoveTheNoise)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_specsub_output.raw", Format, ChannelCount, FrameSampleCount);

    SpectralSubtractionNoiseRemover testee(
        ChannelCount,
        FrameSampleCount,
        createConstantNoiseEstimatorFromFile(resourcesPath + "/noises.txt"),
        Alpha0,
        Gamma,
        Beta);
    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames);
}

TEST(SpectralSubtractionNoiseRemoverTests, removeNoise_alternating_shouldNotCreateGlitch)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;
    constexpr size_t N = 10;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(
        resourcesPath + "/noisy_sounds_alternating_specsub_output.raw",
        Format,
        ChannelCount,
        FrameSampleCount);

    SpectralSubtractionNoiseRemover testee(
        ChannelCount,
        FrameSampleCount,
        createConstantNoiseEstimatorFromFile(resourcesPath + "/noises.txt"),
        Alpha0,
        Gamma,
        Beta);

    PackedAudioFrame<float> inputFrame(ChannelCount, FrameSampleCount);
    PackedAudioFrame<float> expectedOutputFrame(ChannelCount, FrameSampleCount);

    const size_t frameCount = min(inputPcmFrames.size(), expectedOutputPcmFrames.size());
    for (size_t i = 0; i < frameCount; i++)
    {
        if ((i % (N * 2)) < N)
        {
            inputPcmFrames[i].copyTo(inputFrame);
            PackedAudioFrame<float> outputFrame = testee.removeNoise(inputFrame);
            expectedOutputPcmFrames[i].copyTo(expectedOutputFrame);

            expectFrameNear(outputFrame, expectedOutputFrame, AbsError);
        }
        else
        {
            testee.replaceLastFrame(inputFrame);
        }
    }
}
