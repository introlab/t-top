#include <ego_noise_reduction/LogMmseNoiseRemover.h>
#include <ego_noise_reduction/Math.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <cmath>

#include <boost/math/special_functions/expint.hpp>

using namespace introlab;
using namespace boost::math;
using namespace std;


LogMmseNoiseRemover::LogMmseNoiseRemover(
    size_t channelCount,
    size_t frameSampleCount,
    std::shared_ptr<NoiseEstimator> noiseEstimator,
    float alpha,
    float maxAPosterioriSnr,
    float minAPrioriSnr)
    : StftNoiseRemover(channelCount, frameSampleCount, move(noiseEstimator)),
      m_alpha(alpha),
      m_maxAPosterioriSnr(maxAPosterioriSnr),
      m_minAPrioriSnr(minAPrioriSnr)
{
    if (alpha < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Alpha must be greater than or equal to 0.");
    }
    if (maxAPosterioriSnr < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The max a posteriori SNR must be greater than or equal to 0.");
    }
    if (minAPrioriSnr < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The min a priori SNR must be greater than or equal to 0.");
    }

    m_lastInputPowerSpectrum.zeros(m_fftOutputSize, m_channelCount);
    m_noisePowerSpectrum.zeros(m_fftOutputSize);
    m_inputMagnitudeSpectrum.zeros(m_fftOutputSize);
    m_inputPowerSpectrum.zeros(m_fftOutputSize);
    m_aPosterioriSnr.zeros(m_fftOutputSize);
    m_aPosterioriSnrMinus1.zeros(m_fftOutputSize);
    m_aPrioriSnr.zeros(m_fftOutputSize);

    m_A.zeros(m_fftOutputSize);
    m_H.zeros(m_fftOutputSize);
}

LogMmseNoiseRemover::~LogMmseNoiseRemover() {}

void LogMmseNoiseRemover::updateSpectrum(
    size_t channelIndex,
    const arma::cx_fvec& input,
    arma::cx_fvec& output,
    const arma::fvec& noiseMagnitudeSpectrum)
{
    constexpr float EPS = 1e-9;

    m_noisePowerSpectrum = arma::square(noiseMagnitudeSpectrum);
    m_noisePowerSpectrum.transform([=](float v) { return max(v, EPS); });
    m_inputMagnitudeSpectrum = arma::abs(input);
    m_inputPowerSpectrum = arma::square(m_inputMagnitudeSpectrum);

    m_aPosterioriSnr = m_inputPowerSpectrum / m_noisePowerSpectrum;
    m_aPosterioriSnr.transform([=](float v) { return min(v, m_maxAPosterioriSnr); });

    m_aPosterioriSnrMinus1 = m_aPosterioriSnr - 1;
    m_aPosterioriSnrMinus1.transform([=](float v) { return max(v, EPS); });
    m_aPrioriSnr = m_alpha * m_lastInputPowerSpectrum.col(channelIndex) / m_noisePowerSpectrum +
                   (1.f - m_alpha) * m_aPosterioriSnrMinus1;
    m_aPrioriSnr.transform([=](float v) { return max(v, m_minAPrioriSnr); });

    m_A = m_aPrioriSnr / (1.f + m_aPrioriSnr);
    m_H = m_A % m_aPosterioriSnr;
    m_H.transform([](float v) { return -0.5f * expint(-v); });
    m_H = m_A % arma::exp(m_H);

    output = m_H % input;
    m_lastInputPowerSpectrum.col(channelIndex) = arma::square(m_H % m_inputMagnitudeSpectrum);
}
