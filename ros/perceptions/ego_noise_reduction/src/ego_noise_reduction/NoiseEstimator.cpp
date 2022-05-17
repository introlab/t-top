#include <ego_noise_reduction/NoiseEstimator.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

using namespace introlab;
using namespace std;

NoiseEstimator::NoiseEstimator(size_t channelCount, size_t frameSampleCount)
    : m_channelCount(channelCount),
      m_fftOutputSize(frameSampleCount / 2 + 1)
{
    if (m_channelCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The channel count must be greater than 0.");
    }
    if (frameSampleCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The frame sample count must be greater than 0.");
    }
}

NoiseEstimator::~NoiseEstimator() {}

void NoiseEstimator::estimate(
    arma::fvec& noiseMagnitudeSpectrum,
    const arma::cx_fvec& signalSpectrum,
    size_t channelIndex)
{
    if (channelIndex >= m_channelCount)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Channel index is out of range.");
    }
    if (signalSpectrum.n_elem != m_fftOutputSize)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The signal magnitude spectrum does have the right shape. "
                                      "It must be (frameSampleCount/2+1 X channelCount)");
    }
}
