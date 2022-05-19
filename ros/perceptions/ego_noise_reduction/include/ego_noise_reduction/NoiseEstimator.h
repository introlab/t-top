#ifndef EGO_NOISE_REDUCTION_NOISE_ESTIMATOR_H
#define EGO_NOISE_REDUCTION_NOISE_ESTIMATOR_H

#include <armadillo>

class NoiseEstimator
{
    std::size_t m_channelCount;
    std::size_t m_fftOutputSize;

public:
    NoiseEstimator(std::size_t channelCount, std::size_t frameSampleCount);
    virtual ~NoiseEstimator();

    virtual void reset() = 0;
    virtual void
        estimate(arma::fvec& noiseMagnitudeSpectrum, const arma::cx_fvec& signalSpectrum, std::size_t channelIndex);

    std::size_t channelCount() const;
    std::size_t fftOutputSize() const;
};

inline std::size_t NoiseEstimator::channelCount() const
{
    return m_channelCount;
}

inline std::size_t NoiseEstimator::fftOutputSize() const
{
    return m_fftOutputSize;
}

#endif
