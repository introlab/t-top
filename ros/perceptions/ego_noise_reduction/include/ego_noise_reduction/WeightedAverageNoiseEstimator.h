#ifndef EGO_NOISE_REDUCTION_WEIGHTED_AVERAGE_NOISE_ESTIMATOR_H
#define EGO_NOISE_REDUCTION_WEIGHTED_AVERAGE_NOISE_ESTIMATOR_H

#include <ego_noise_reduction/NoiseEstimator.h>

class WeightedAverageNoiseEstimator : public NoiseEstimator
{
    float m_epsilonSquared;
    float m_alpha;
    float m_delta;

    bool m_isReset;

    arma::fmat m_lastNoiseMagnitudeSpectrum;
    arma::fmat m_lastNoiseMagnitudeSpectrumVariance;

    arma::fvec m_signalMagnitudeSpectrum;

public:
    WeightedAverageNoiseEstimator(
        std::size_t channelCount,
        std::size_t frameSampleCount,
        float epsilon,
        float alpha,
        float delta);
    ~WeightedAverageNoiseEstimator() override;

    void reset() override;
    void estimate(arma::fvec& noiseMagnitudeSpectrum, const arma::cx_fvec& signalSpectrum, std::size_t channelIndex)
        override;

private:
    void resetBuffers();
};

#endif
