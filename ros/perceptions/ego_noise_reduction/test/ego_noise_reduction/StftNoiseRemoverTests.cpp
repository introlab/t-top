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
    DummyStftNoiseRemover(size_t channelCount, size_t frameSampleCount) : StftNoiseRemover(channelCount, frameSampleCount)
    {
    }

    ~DummyStftNoiseRemover() override
    {
    }

    DECLARE_NOT_COPYABLE(DummyStftNoiseRemover);
    DECLARE_NOT_MOVABLE(DummyStftNoiseRemover);

protected:
    void updateSpectrum(size_t channelIndex, const arma::cx_fvec& input, arma::cx_fvec& output, const arma::fvec& noiseMagnitudeSpectrum) override
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

TEST(StftNoiseRemoverTests, replaceLastFrame_invalidChannelCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    AudioFrame<float> frame(ChannelCount - 1, FrameSampleCount);
    EXPECT_THROW(testee.replaceLastFrame(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, replaceLastFrame_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    AudioFrame<float> frame(ChannelCount, FrameSampleCount - 1);
    EXPECT_THROW(testee.replaceLastFrame(frame), NotSupportedException);
}

TEST(StftNoiseRemoverTests, replaceLastFrame_shouldSetTheBuffers)
{
    constexpr size_t ChannelCount = 1;
    constexpr size_t FrameSampleCount = 4;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    AudioFrame<float> inputFrame(ChannelCount, FrameSampleCount);
    inputFrame[0] = 1.f;
    inputFrame[1] = 2.f;
    inputFrame[2] = 3.f;
    inputFrame[3] = 4.f;
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);

    testee.replaceLastFrame(inputFrame);
    AudioFrame<float> outputFrame = testee.removeNoise(inputFrame, noiseMagnitudeSpectrum);

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

    AudioFrame<float> frame(ChannelCount - 1, FrameSampleCount);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum), NotSupportedException);
}

TEST(StftNoiseRemoverTests, removeNoise_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    AudioFrame<float> frame(ChannelCount, FrameSampleCount - 1);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum), NotSupportedException);
}

TEST(StftNoiseRemoverTests, removeNoise_invalidNoiseMagnitudeSpectrum_shouldThrowNotSupportedException)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    AudioFrame<float> frame(ChannelCount, FrameSampleCount);
    arma::fmat noiseMagnitudeSpectrum1 = arma::zeros<arma::fmat>(FrameSampleCount / 2, ChannelCount);
    arma::fmat noiseMagnitudeSpectrum2 = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount - 1);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum1), NotSupportedException);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum2), NotSupportedException);
}

TEST(StftNoiseRemoverTests, removeNoise_shouldReturnTheSameSignal)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameSampleCount = 2048;
    constexpr PcmAudioFrameFormat FORMAT = PcmAudioFrameFormat::Signed32;

    DummyStftNoiseRemover testee(ChannelCount, FrameSampleCount);

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", FORMAT, ChannelCount, FrameSampleCount);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds_zero_output.raw", FORMAT, ChannelCount, FrameSampleCount);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FrameSampleCount / 2 + 1, ChannelCount);

    testNoiseReduction(testee, inputPcmFrames, expectedOutputPcmFrames, noiseMagnitudeSpectrum);
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
