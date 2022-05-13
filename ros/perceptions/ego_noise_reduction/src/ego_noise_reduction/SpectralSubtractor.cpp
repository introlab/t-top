#include <ego_noise_reduction/SpectralSubtractor.h>
#include <ego_noise_reduction/Math.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <cmath>

using namespace introlab;
using namespace std;

SpectralSubtractor::SpectralSubtractor(
    size_t channelCount,
    size_t frameSampleCount,
    float alpha0,
    float gamma,
    float beta)
    : m_channelCount(channelCount),
      m_frameSampleCount(frameSampleCount),
      m_step(m_frameSampleCount / 2),
      m_alpha0(alpha0),
      m_gamma(gamma),
      m_beta(beta)
{
    if (m_channelCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The channel count must be greater than 0.");
    }
    if ((m_frameSampleCount % 2) != 0 || m_frameSampleCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The frame sample count must be greater than 0 and even.");
    }
    if (m_alpha0 < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The alpha must be greater than or equal to 0.");
    }
    if (gamma < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The alpha must be greater than or equal to 0.");
    }
    if (beta < 0.f)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The alpha must be greater than or equal to 0.");
    }

    size_t bufferColCount = 3 * m_step;
    m_inputBuffer.zeros(bufferColCount, m_channelCount);
    m_outputBuffer.zeros(bufferColCount, m_channelCount);
    m_output.zeros(m_frameSampleCount, m_channelCount);

    m_window = arma::sqrt(hann<arma::fvec>(m_frameSampleCount));
    m_singleOutput.zeros(m_frameSampleCount);

    m_fftInput.zeros(m_frameSampleCount);
    m_fftOutput.zeros(m_frameSampleCount / 2 + 1);
    m_fftPlan = fftwf_plan_dft_r2c_1d(
        m_frameSampleCount,
        m_fftInput.memptr(),
        reinterpret_cast<fftwf_complex*>(m_fftOutput.memptr()),
        FFTW_ESTIMATE);

    m_fftMagnitude.zeros(m_fftOutput.n_elem);
    m_fftAngle.zeros(m_fftOutput.n_elem);

    m_ifftInput.zeros(m_fftOutput.n_elem);
    m_ifftOutput.zeros(m_fftInput.n_elem);
    m_ifftPlan = fftwf_plan_dft_c2r_1d(
        m_frameSampleCount,
        reinterpret_cast<fftwf_complex*>(m_ifftInput.memptr()),
        m_ifftOutput.memptr(),
        FFTW_ESTIMATE);
}

SpectralSubtractor::~SpectralSubtractor()
{
    fftwf_destroy_plan(m_fftPlan);
    fftwf_destroy_plan(m_ifftPlan);
}

AudioFrame<float>
    SpectralSubtractor::removeNoise(const AudioFrame<float>& input, const arma::fmat& noiseMagnitudeSpectrum)
{
    if (input.channelCount() != m_channelCount || input.sampleCount() != m_frameSampleCount)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The input frame does not match the constructor parameters.");
    }
    if (noiseMagnitudeSpectrum.n_rows != m_fftOutput.n_elem || noiseMagnitudeSpectrum.n_cols != m_channelCount)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The noise magnitude spectrum does have the right shape. "
                                      "It must be (frameSampleCount/2+1 X channelCount)");
    }

    const arma::fmat inputMat(const_cast<float*>(input.data()), m_frameSampleCount, m_channelCount, false, true);

    m_inputBuffer.rows(m_step, m_inputBuffer.n_rows - 1) = inputMat;
    m_outputBuffer.rows(m_step, m_outputBuffer.n_rows - 1).zeros();

    for (size_t c = 0; c < m_channelCount; c++)
    {
        removeNoise(c, noiseMagnitudeSpectrum);
    }

    m_output = m_outputBuffer.rows(0, m_frameSampleCount - 1);
    m_inputBuffer.rows(0, m_step - 1) = m_inputBuffer.rows(m_frameSampleCount, m_inputBuffer.n_rows - 1);
    m_outputBuffer.rows(0, m_step - 1) = m_outputBuffer.rows(m_frameSampleCount, m_outputBuffer.n_rows - 1);


    return AudioFrame<float>(m_channelCount, m_frameSampleCount, m_output.memptr());
}

void SpectralSubtractor::removeNoise(size_t c, const arma::fmat& noiseMagnitudeSpectrum)
{
    removeNoise(
        m_singleOutput,
        m_inputBuffer(arma::span(0, m_frameSampleCount - 1), c) % m_window,
        noiseMagnitudeSpectrum.col(c));
    m_outputBuffer(arma::span(0, m_frameSampleCount - 1), c) += m_singleOutput % m_window;

    removeNoise(
        m_singleOutput,
        m_inputBuffer(arma::span(m_step, m_inputBuffer.n_rows - 1), c) % m_window,
        noiseMagnitudeSpectrum.col(c));
    m_outputBuffer(arma::span(m_step, m_outputBuffer.n_rows - 1), c) += m_singleOutput % m_window;
}


void SpectralSubtractor::removeNoise(
    arma::fvec& output,
    const arma::fvec& input,
    const arma::fvec& noiseMagnitudeSpectrum)
{
    m_fftInput = input;
    fftwf_execute(m_fftPlan);

    m_fftMagnitude = arma::abs(m_fftOutput);
    m_fftAngle = arma::arg(m_fftOutput);

    float snr = 20 * log10(arma::sum(m_fftMagnitude) / arma::sum(noiseMagnitudeSpectrum));
    float alpha = max(1.f, min(m_alpha0 - m_gamma * snr, m_alpha0));
    float alphaBetaSum = alpha + m_beta;

    float* fftMagnitudeData = m_fftMagnitude.memptr();
    const float* fftAngleData = m_fftAngle.memptr();
    const float* noiseMagnitudeSpectrumData = noiseMagnitudeSpectrum.memptr();
    complex<float>* ifftInputData = m_ifftInput.memptr();

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
        ifftInputData[i] = polar(fftMagnitudeData[i], fftAngleData[i]);
    }

    fftwf_execute(m_ifftPlan);
    output = m_ifftOutput / static_cast<float>(m_frameSampleCount);
}
