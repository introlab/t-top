#include <ego_noise_reduction/WeightedAverageNoiseEstimator.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <cmath>

using namespace std;

WeightedAverageNoiseEstimator::WeightedAverageNoiseEstimator(
    size_t channelCount,
    size_t frameSampleCount,
    float epsilon,
    float alpha,
    float delta)
    : NoiseEstimator(channelCount, frameSampleCount),
      m_epsilonSquared(epsilon * epsilon),
      m_alpha(alpha),
      m_delta(delta),
      m_isReset(false)
{
    if (epsilon <= 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The epsilon must be greater than 0.");
    }
    if (alpha < 0.f || alpha > 1.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The alpha must be between than 0 and 1.");
    }
    if (delta < 0.f || delta > 1.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The delta must be between than 0 and 1.");
    }

    resetBuffers();
    m_signalMagnitudeSpectrum.zeros(fftOutputSize());
}

WeightedAverageNoiseEstimator::~WeightedAverageNoiseEstimator() {}

void WeightedAverageNoiseEstimator::reset()
{
    resetBuffers();
}

void WeightedAverageNoiseEstimator::estimate(
    arma::fvec& noiseMagnitudeSpectrum,
    const arma::cx_fvec& signalSpectrum,
    size_t channelIndex)
{
    NoiseEstimator::estimate(noiseMagnitudeSpectrum, signalSpectrum, channelIndex);
    m_isReset = false;

    m_signalMagnitudeSpectrum = arma::abs(signalSpectrum);

    for (size_t i = 0; i < fftOutputSize(); i++)
    {
        float lastNoiseMagnitudeSpectrum = m_lastNoiseMagnitudeSpectrum.at(i, channelIndex);
        float lastNoiseMagnitudeSpectrumVariance = m_lastNoiseMagnitudeSpectrumVariance.at(i, channelIndex);
        float signalMagnitudeSpectrum = m_signalMagnitudeSpectrum[i];

        float absDiff = abs(signalMagnitudeSpectrum - lastNoiseMagnitudeSpectrum);
        if ((absDiff * absDiff) < (m_epsilonSquared * lastNoiseMagnitudeSpectrumVariance))
        {
            float noiseMagnitude = m_alpha * lastNoiseMagnitudeSpectrum + (1.f - m_alpha) * signalMagnitudeSpectrum;
            m_lastNoiseMagnitudeSpectrum.at(i, channelIndex) = noiseMagnitude;

            float diff = signalMagnitudeSpectrum - noiseMagnitude;
            m_lastNoiseMagnitudeSpectrumVariance.at(i, channelIndex) =
                m_delta * lastNoiseMagnitudeSpectrumVariance + (1.f - m_delta) * diff * diff;
        }
    }

    noiseMagnitudeSpectrum = m_lastNoiseMagnitudeSpectrum.col(channelIndex);
}

void WeightedAverageNoiseEstimator::resetBuffers()
{
    if (m_isReset)
    {
        return;
    }

    m_lastNoiseMagnitudeSpectrum.zeros(fftOutputSize(), channelCount());
    m_lastNoiseMagnitudeSpectrumVariance.ones(fftOutputSize(), channelCount());
    m_isReset = true;
}
