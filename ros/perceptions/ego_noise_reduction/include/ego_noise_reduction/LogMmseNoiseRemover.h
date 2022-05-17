#ifndef EGO_NOISE_REDUCTION_LOG_MMSE_NOISE_REMOVER_H
#define EGO_NOISE_REDUCTION_LOG_MMSE_NOISE_REMOVER_H

#include <ego_noise_reduction/StftNoiseRemover.h>

class LogMmseNoiseRemover : public StftNoiseRemover
{
    float m_alpha;
    float m_maxAPosterioriSnr;
    float m_minAPrioriSnr;

    arma::fmat m_lastInputPowerSpectrum;
    arma::fvec m_noisePowerSpectrum;
    arma::fvec m_inputMagnitudeSpectrum;
    arma::fvec m_inputPowerSpectrum;
    arma::fvec m_aPosterioriSnr;
    arma::fvec m_aPosterioriSnrMinus1;
    arma::fvec m_aPrioriSnr;

    arma::fvec m_A;
    arma::fvec m_H;

public:
    LogMmseNoiseRemover(
        std::size_t channelCount,
        std::size_t frameSampleCount,
        std::shared_ptr<NoiseEstimator> noiseEstimator,
        float alpha,
        float maxAPosterioriSnr,
        float minAPrioriSnr);
    ~LogMmseNoiseRemover() override;

    DECLARE_NOT_COPYABLE(LogMmseNoiseRemover);
    DECLARE_NOT_MOVABLE(LogMmseNoiseRemover);

protected:
    void updateSpectrum(
        std::size_t channelIndex,
        const arma::cx_fvec& input,
        arma::cx_fvec& output,
        const arma::fvec& noiseMagnitudeSpectrum) override;
};

#endif
