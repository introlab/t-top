#ifndef EGO_NOISE_REDUCTION_NOISE_REMOVER_H
#define EGO_NOISE_REDUCTION_NOISE_REMOVER_H

#include <ego_noise_reduction/NoiseEstimator.h>

#include <MusicBeatDetector/Utils/Data/PackedAudioFrame.h>
#include <MusicBeatDetector/Utils/ClassMacro.h>

#include <armadillo>
#include <fftw3.h>

#include <memory>

class StftNoiseRemover
{
public:
    enum class Type
    {
        SpectralSubtraction,
        LogMmse
    };

protected:
    const std::size_t m_channelCount;
    const std::size_t m_frameSampleCount;
    const std::size_t m_step;
    const std::size_t m_fftOutputSize;

private:
    std::shared_ptr<NoiseEstimator> m_noiseEstimator;
    arma::fvec m_noiseMagnitudeSpectrum;

    arma::fmat m_inputBuffer;
    arma::fmat m_outputBuffer;
    arma::fmat m_output;

    arma::fvec m_window;
    arma::fvec m_singleOutput;

    arma::fvec m_fftInput;
    arma::cx_fvec m_fftOutput;
    fftwf_plan m_fftPlan;

    arma::cx_fvec m_ifftInput;
    arma::fvec m_ifftOutput;
    fftwf_plan m_ifftPlan;

public:
    StftNoiseRemover(
        std::size_t channelCount,
        std::size_t frameSampleCount,
        std::shared_ptr<NoiseEstimator> noiseEstimator);
    virtual ~StftNoiseRemover();

    DECLARE_NOT_COPYABLE(StftNoiseRemover);
    DECLARE_NOT_MOVABLE(StftNoiseRemover);

    void replaceLastFrame(const introlab::PackedAudioFrame<float>& input);

    introlab::PackedAudioFrame<float> removeNoise(const introlab::PackedAudioFrame<float>& input);

private:
    void removeNoise(std::size_t channelIndex);
    void removeNoise(std::size_t channelIndex, arma::fvec& output, const arma::fvec& input);

protected:
    virtual void updateSpectrum(
        std::size_t channelIndex,
        const arma::cx_fvec& input,
        arma::cx_fvec& output,
        const arma::fvec& noiseMagnitudeSpectrum) = 0;

public:
    static Type parseType(const std::string& backend);
};

#endif
