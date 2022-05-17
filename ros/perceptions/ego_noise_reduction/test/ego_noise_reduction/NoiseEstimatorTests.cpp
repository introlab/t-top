#include <ego_noise_reduction/NoiseEstimator.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

using namespace introlab;
using namespace std;

class DummyNoiseEstimator : public NoiseEstimator
{
public:
    DummyNoiseEstimator(size_t channelCount, size_t frameSampleCount) : NoiseEstimator(channelCount, frameSampleCount)
    {
    }
    ~DummyNoiseEstimator() override {}

    void reset() override {}
};

TEST(NoiseEstimatorTests, constructor_channelCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyNoiseEstimator(0, 1), NotSupportedException);
}

TEST(NoiseEstimatorTests, constructor_frameSampleCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(DummyNoiseEstimator(1, 0), NotSupportedException);
}

TEST(NoiseEstimatorTests, constructor_shouldSetTheRightValues)
{
    DummyNoiseEstimator testee(1, 5);

    EXPECT_EQ(testee.channelCount(), 1);
    EXPECT_EQ(testee.fftOutputSize(), 3);
}

TEST(NoiseEstimatorTests, estimate_invalidSignalSpectrum_shouldThrowNotSupportedException)
{
    DummyNoiseEstimator testee(1, 5);

    arma::fvec noiseMagnitudeSpectrum;
    arma::cx_fvec signalSpectrum;
    EXPECT_THROW(testee.estimate(noiseMagnitudeSpectrum, signalSpectrum, 0), NotSupportedException);
}

TEST(NoiseEstimatorTests, estimate_invalidChannelIndex_shouldThrowNotSupportedException)
{
    DummyNoiseEstimator testee(1, 5);

    arma::fvec noiseMagnitudeSpectrum;
    arma::cx_fvec signalSpectrum;
    signalSpectrum.zeros(testee.fftOutputSize());

    EXPECT_THROW(testee.estimate(noiseMagnitudeSpectrum, signalSpectrum, 1), NotSupportedException);
}
