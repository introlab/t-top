#include <ego_noise_reduction/StftNoiseRemover.h>
#include <ego_noise_reduction/Math.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>
#include <MusicBeatDetector/Utils/Exception/InvalidValueException.h>

#include <cmath>

using namespace introlab;
using namespace std;

StftNoiseRemover::StftNoiseRemover(size_t channelCount, size_t frameSampleCount)
    : m_channelCount(channelCount),
      m_frameSampleCount(frameSampleCount),
      m_step(m_frameSampleCount / 2),
      m_fftOutputSize(m_frameSampleCount / 2 + 1)
{
    if (m_channelCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The channel count must be greater than 0.");
    }
    if ((m_frameSampleCount % 2) != 0 || m_frameSampleCount == 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The frame sample count must be greater than 0 and even.");
    }

    size_t bufferColCount = 3 * m_step;
    m_inputBuffer.zeros(bufferColCount, m_channelCount);
    m_outputBuffer.zeros(bufferColCount, m_channelCount);
    m_output.zeros(m_frameSampleCount, m_channelCount);

    m_window = arma::sqrt(hann<arma::fvec>(m_frameSampleCount));
    m_singleOutput.zeros(m_frameSampleCount);

    m_fftInput.zeros(m_frameSampleCount);
    m_fftOutput.zeros(m_fftOutputSize);
    m_fftPlan = fftwf_plan_dft_r2c_1d(
        m_frameSampleCount,
        m_fftInput.memptr(),
        reinterpret_cast<fftwf_complex*>(m_fftOutput.memptr()),
        FFTW_ESTIMATE);

    m_ifftInput.zeros(m_fftOutputSize);
    m_ifftOutput.zeros(m_fftInput.n_elem);
    m_ifftPlan = fftwf_plan_dft_c2r_1d(
        m_frameSampleCount,
        reinterpret_cast<fftwf_complex*>(m_ifftInput.memptr()),
        m_ifftOutput.memptr(),
        FFTW_ESTIMATE);
}

StftNoiseRemover::~StftNoiseRemover()
{
    fftwf_destroy_plan(m_fftPlan);
    fftwf_destroy_plan(m_ifftPlan);
}

void StftNoiseRemover::replaceLastFrame(const introlab::AudioFrame<float>& input)
{
    if (input.channelCount() != m_channelCount || input.sampleCount() != m_frameSampleCount)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The input frame does not match the constructor parameters.");
    }

    const arma::fmat inputMat(const_cast<float*>(input.data()), m_frameSampleCount, m_channelCount, false, true);

    m_inputBuffer.rows(0, m_step - 1) = inputMat.rows(m_step, m_frameSampleCount - 1);
    m_outputBuffer.rows(0, m_step - 1) =
        inputMat.rows(m_step, m_frameSampleCount - 1).eval().each_col() % m_window.rows(m_step, m_frameSampleCount - 1);
}

AudioFrame<float>
    StftNoiseRemover::removeNoise(const AudioFrame<float>& input, const arma::fmat& noiseMagnitudeSpectrum)
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

    for (size_t channelIndex = 0; channelIndex < m_channelCount; channelIndex++)
    {
        removeNoise(channelIndex, noiseMagnitudeSpectrum);
    }

    m_output = m_outputBuffer.rows(0, m_frameSampleCount - 1);
    m_inputBuffer.rows(0, m_step - 1) = m_inputBuffer.rows(m_frameSampleCount, m_inputBuffer.n_rows - 1);
    m_outputBuffer.rows(0, m_step - 1) = m_outputBuffer.rows(m_frameSampleCount, m_outputBuffer.n_rows - 1);


    return AudioFrame<float>(m_channelCount, m_frameSampleCount, m_output.memptr());
}

void StftNoiseRemover::removeNoise(size_t channelIndex, const arma::fmat& noiseMagnitudeSpectrum)
{
    removeNoise(
        channelIndex,
        m_singleOutput,
        m_inputBuffer(arma::span(0, m_frameSampleCount - 1), channelIndex) % m_window,
        noiseMagnitudeSpectrum.col(channelIndex));
    m_outputBuffer(arma::span(0, m_frameSampleCount - 1), channelIndex) += m_singleOutput % m_window;

    removeNoise(
        channelIndex,
        m_singleOutput,
        m_inputBuffer(arma::span(m_step, m_inputBuffer.n_rows - 1), channelIndex) % m_window,
        noiseMagnitudeSpectrum.col(channelIndex));
    m_outputBuffer(arma::span(m_step, m_outputBuffer.n_rows - 1), channelIndex) += m_singleOutput % m_window;
}

void StftNoiseRemover::removeNoise(
    size_t channelIndex,
    arma::fvec& output,
    const arma::fvec& input,
    const arma::fvec& noiseMagnitudeSpectrum)
{
    m_fftInput = input;
    fftwf_execute(m_fftPlan);

    updateSpectrum(channelIndex, m_fftOutput, m_ifftInput, noiseMagnitudeSpectrum);

    fftwf_execute(m_ifftPlan);
    output = m_ifftOutput / static_cast<float>(m_frameSampleCount);
}

StftNoiseRemover::Type StftNoiseRemover::parseType(const std::string& type)
{
    if (type == "spectral_subtraction")
    {
        return StftNoiseRemover::Type::SpectralSubtraction;
    }
    else if (type == "log_mmse")
    {
        return StftNoiseRemover::Type::LogMmse;
    }

    THROW_INVALID_VALUE_EXCEPTION("type", type);
}
