#include <ego_noise_reduction/StftNoiseRemover.h>

#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>
#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>
#include <MusicBeatDetector/Utils/Exception/InvalidValueException.h>

#include "TestUtils.h"

#include <gtest/gtest.h>

#include <vector>

using namespace introlab;
using namespace std;

constexpr float AbsError = 0.001f;

class DummyStftNoiseRemover : public StftNoiseRemover
{
public:
    DummyStftNoiseRemover(size_t channelCount, size_t frameSampleCount)
        : StftNoiseRemover(
              channelCount,
              frameSampleCount,
              createZeroConstantNoiseEstimator(channelCount, frameSampleCount))
    {
    }

    DummyStftNoiseRemover(
        size_t channelCount,
        size_t frameSampleCount,
        size_t neChannelCount,
        size_t neFrameSampleCount)
        : StftNoiseRemover(
              channelCount,
              frameSampleCount,
              createZeroConstantNoiseEstimator(neChannelCount, neFrameSampleCount))
    {
    }

    ~DummyStftNoiseRemover() override {}

    DECLARE_NOT_COPYABLE(DummyStftNoiseRemover);
    DECLARE_NOT_MOVABLE(DummyStftNoiseRemover);

    size_t channelCount() const { return m_channelCount; }
    size_t frameSampleCount() const { return m_frameSampleCount; }
    size_t step() const { return m_step; }
    size_t fftOutputSize() { return m_fftOutputSize; };

protected:
    void updateSpectrum(
        size_t channelIndex,
        const arma::cx_fvec& input,
        arma::cx_fvec& output,
        const arma::fvec& noiseMagnitudeSpectrum) override
    {
        output = input;
    }
};

TEST(StftNoiseRemoverTests, constructor_channelCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyStftNoiseRemover(0, 1), NotSupportedException);
}

TEST(StftNoiseRemoverTests, constructor_frameSampleCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyStftNoiseRemover(1, 0), NotSupportedException);
}

TEST(StftNoiseRemoverTests, constructor_oddFrameSampleCount_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyStftNoiseRemover(1, 1), NotSupportedException);
}

TEST(StftNoiseRemoverTests, constructor_invalidNoiseEstimatorChannelCount_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyStftNoiseRemover(1, 4, 2, 4), NotSupportedException);
}

TEST(StftNoiseRemoverTests, constructor_invalidNoiseEstimatorFrameSampleCount_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyStftNoiseRemover(1, 4, 1, 6), NotSupportedException);
}

TEST(StftNoiseRemoverTests, constructor_shouldSetTheRightValues)
{
    DummyStftNoiseRemover testee(1, 4);

    EXPECT_EQ(testee.channelCount(), 1);
    EXPECT_EQ(testee.frameSampleCount(), 4);
    EXPECT_EQ(testee.step(), 2);
    EXPECT_EQ(testee.fftOutputSize(), 3);
}

TEST(StftNoiseRemoverTests, replaceLastFrame_invalidChannelCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    PackedAudioFrame<float> frame(ChannelCount - 1, FrameSampleCount);
    EXPECT_THROW(testee.replaceLastFrame(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, replaceLastFrame_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    PackedAudioFrame<float> frame(ChannelCount, FrameSampleCount - 1);
    EXPECT_THROW(testee.replaceLastFrame(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, replaceLastFrame_shouldSetTheBuffers)
{
    constexpr size_t ChannelCount = 1;
    constexpr size_t FrameSampleCount = 4;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    PackedAudioFrame<float> inputFrame(ChannelCount, FrameSampleCount);
    inputFrame[0] = 1.f;
    inputFrame[1] = 2.f;
    inputFrame[2] = 3.f;
    inputFrame[3] = 4.f;

    testee.replaceLastFrame(inputFrame);
    PackedAudioFrame<float> outputFrame = testee.removeNoise(inputFrame);

    EXPECT_NEAR(outputFrame[0], 2.598f, AbsError);
    EXPECT_NEAR(outputFrame[1], 3.f, AbsError);
    EXPECT_NEAR(outputFrame[2], 0.75f, AbsError);
    EXPECT_NEAR(outputFrame[3], 1.5f, AbsError);
}

TEST(StftNoiseRemoverTests, removeNoise_invalidChannelCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    PackedAudioFrame<float> frame(ChannelCount - 1, FrameSampleCount);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);
    EXPECT_THROW(testee.removeNoise(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, removeNoise_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    PackedAudioFrame<float> frame(ChannelCount, FrameSampleCount - 1);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);
    EXPECT_THROW(testee.removeNoise(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, removeNoise_shouldReturnTheSameSignal)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Signed32;

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", Format, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames =
        getPcmAudioFrames(resourcesPath + "/noisy_sounds_zero_output.raw", Format, ChannelCount, FrameSampleCount);

    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);
    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames);
}

TEST(StftNoiseRemoverTests, parseType_invalidType_shouldThrowInvalidValueException)
{
    EXPECT_THROW(StftNoiseRemover::parseType("asb"), InvalidValueException);
}

TEST(StftNoiseRemoverTests, parseType_spectralSubtraction_shouldReturnSpectralSubtraction)
{
    EXPECT_EQ(StftNoiseRemover::parseType("spectral_subtraction"), StftNoiseRemover::Type::SpectralSubtraction);
}

TEST(StftNoiseRemoverTests, parseType_logMmse_shouldReturnLogMmse)
{
    EXPECT_EQ(StftNoiseRemover::parseType("log_mmse"), StftNoiseRemover::Type::LogMmse);
}
