#ifndef EGO_NOISE_REDUCTION_SPECTRAL_SUBTRACTION_NOISE_REMOVER_H
#define EGO_NOISE_REDUCTION_SPECTRAL_SUBTRACTION_NOISE_REMOVER_H

#include <ego_noise_reduction/StftNoiseRemover.h>

class SpectralSubtractionNoiseRemover : public StftNoiseRemover
{
    float m_alpha0;
    float m_gamma;
    float m_beta;

    arma::fvec m_fftMagnitude;
    arma::fvec m_fftAngle;

public:
    SpectralSubtractionNoiseRemover(
        std::size_t channelCount,
        std::size_t frameSampleCount,
        std::shared_ptr<NoiseEstimator> noiseEstimator,
        float alpha0,
        float gamma,
        float beta);
    ~SpectralSubtractionNoiseRemover() override;

    DECLARE_NOT_COPYABLE(SpectralSubtractionNoiseRemover);
    DECLARE_NOT_MOVABLE(SpectralSubtractionNoiseRemover);

protected:
    void updateSpectrum(
        std::size_t channelIndex,
        const arma::cx_fvec& input,
        arma::cx_fvec& output,
        const arma::fvec& noiseMagnitudeSpectrum) override;
};

#endif
