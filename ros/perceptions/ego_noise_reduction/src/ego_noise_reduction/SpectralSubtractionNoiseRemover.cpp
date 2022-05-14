#include <ego_noise_reduction/SpectralSubtractionNoiseRemover.h>
#include <ego_noise_reduction/Math.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <cmath>

using namespace introlab;
using namespace std;

SpectralSubtractionNoiseRemover::SpectralSubtractionNoiseRemover(
    size_t channelCount,
    size_t frameSampleCount,
    float alpha0,
    float gamma,
    float beta)
    : StftNoiseRemover(channelCount, frameSampleCount),
      m_alpha0(alpha0),
      m_gamma(gamma),
      m_beta(beta)
{
    if (m_alpha0 < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Alpha0 must be greater than or equal to 0.");
    }
    if (gamma < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Gamma must be greater than or equal to 0.");
    }
    if (beta < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Beta must be greater than or equal to 0.");
    }

    m_fftMagnitude.zeros(m_fftOutputSize);
    m_fftAngle.zeros(m_fftOutputSize);
}

SpectralSubtractionNoiseRemover::~SpectralSubtractionNoiseRemover() {}

void SpectralSubtractionNoiseRemover::updateSpectrum(
    size_t channelIndex,
    const arma::cx_fvec& input,
    arma::cx_fvec& output,
    const arma::fvec& noiseMagnitudeSpectrum)
{
    m_fftMagnitude = arma::abs(input);
    m_fftAngle = arma::arg(input);

    float snr = 20 * log10(arma::sum(m_fftMagnitude) / arma::sum(noiseMagnitudeSpectrum));
    float alpha = max(1.f, min(m_alpha0 - m_gamma * snr, m_alpha0));
    float alphaBetaSum = alpha + m_beta;

    float* fftMagnitudeData = m_fftMagnitude.memptr();
    const float* fftAngleData = m_fftAngle.memptr();
    const float* noiseMagnitudeSpectrumData = noiseMagnitudeSpectrum.memptr();
    complex<float>* outputData = output.memptr();

    for (size_t i = 0; i < m_fftMagnitude.n_elem; i++)
    {
        if (fftMagnitudeData[i] > alphaBetaSum * noiseMagnitudeSpectrumData[i])
        {
            fftMagnitudeData[i] -= alpha * noiseMagnitudeSpectrumData[i];
        }
        else
        {
            fftMagnitudeData[i] = m_beta * noiseMagnitudeSpectrumData[i];
        }
        outputData[i] = polar(fftMagnitudeData[i], fftAngleData[i]);
    }
}
