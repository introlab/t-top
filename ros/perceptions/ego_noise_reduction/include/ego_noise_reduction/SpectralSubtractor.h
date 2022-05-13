#ifndef EGO_NOISE_REDUCTION_SPECTRAL_SUBTRACTOR_H
#define EGO_NOISE_REDUCTION_SPECTRAL_SUBTRACTOR_H

#include <MusicBeatDetector/Utils/Data/AudioFrame.h>
#include <MusicBeatDetector/Utils/ClassMacro.h>

#include <armadillo>
#include <fftw3.h>

class SpectralSubtractor
{
    std::size_t m_channelCount;
    std::size_t m_frameSampleCount;
    std::size_t m_step;

    float m_alpha0;
    float m_gamma;
    float m_beta;

    arma::fmat m_inputBuffer;
    arma::fmat m_outputBuffer;
    arma::fmat m_output;

    arma::fvec m_window;
    arma::fvec m_singleOutput;

    arma::fvec m_fftInput;
    arma::cx_fvec m_fftOutput;
    fftwf_plan m_fftPlan;

    arma::fvec m_fftMagnitude;
    arma::fvec m_fftAngle;

    arma::cx_fvec m_ifftInput;
    arma::fvec m_ifftOutput;
    fftwf_plan m_ifftPlan;

public:
    SpectralSubtractor(std::size_t channelCount, std::size_t frameSampleCount, float alpha0, float gamma, float beta);
    virtual ~SpectralSubtractor();

    DECLARE_NOT_COPYABLE(SpectralSubtractor);
    DECLARE_NOT_MOVABLE(SpectralSubtractor);

    introlab::AudioFrame<float>
        removeNoise(const introlab::AudioFrame<float>& input, const arma::fmat& noiseMagnitudeSpectrum);

private:
    void removeNoise(std::size_t c, const arma::fmat& noiseMagnitudeSpectrum);
    void removeNoise(arma::fvec& output, const arma::fvec& input, const arma::fvec& noiseMagnitudeSpectrum);
};

#endif
