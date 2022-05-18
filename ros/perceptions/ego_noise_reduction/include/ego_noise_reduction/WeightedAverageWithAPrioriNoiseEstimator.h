#ifndef EGO_NOISE_REDUCTION_WEIGHTED_AVERAGE_WITH_NOISE_ESTIMATOR_NOISE_ESTIMATOR_H
#define EGO_NOISE_REDUCTION_WEIGHTED_AVERAGE_WITH_NOISE_ESTIMATOR_NOISE_ESTIMATOR_H

#include <ego_noise_reduction/NoiseEstimator.h>
#include <ego_noise_reduction/WeightedAverageNoiseEstimator.h>
#include <ego_noise_reduction/NoiseMagnitudeDatabase.h>

#include <string>
#include <cmath>

constexpr int MinSpeed = 20;
constexpr int MaxTorsoSpeed = 80;
constexpr int MaxHeadSpeed = 265;
constexpr int SpeedStep = 5;

constexpr int MinOrientation = 0;
constexpr int MaxOrientation = 350;
constexpr int OrientationStep = 10;

class WeightedAverageWithAPrioriNoiseEstimator : public NoiseEstimator
{
    WeightedAverageNoiseEstimator m_noiseEstimator;

    std::size_t m_magnitudeSize;
    NoiseMagnitudeDatabase m_torsoDatabase;
    NoiseMagnitudeDatabase m_headDatabaseId1;
    NoiseMagnitudeDatabase m_headDatabaseId2;
    NoiseMagnitudeDatabase m_headDatabaseId3;
    NoiseMagnitudeDatabase m_headDatabaseId4;
    NoiseMagnitudeDatabase m_headDatabaseId5;
    NoiseMagnitudeDatabase m_headDatabaseId6;

    int m_torsoSpeed;
    int m_headSpeedId1;
    int m_headSpeedId2;
    int m_headSpeedId3;
    int m_headSpeedId4;
    int m_headSpeedId5;
    int m_headSpeedId6;

    int m_orientation;

    arma::fvec m_estimatedNoiseMagnitudeSpectrum;
    arma::fvec m_aPrioriNoiseMagnitudeSpectrum;

public:
    WeightedAverageWithAPrioriNoiseEstimator(
        std::size_t channelCount,
        std::size_t frameSampleCount,
        std::size_t samplingFrequency,
        float epsilon,
        float alpha,
        float delta,
        const std::string& noiseDirectory);
    ~WeightedAverageWithAPrioriNoiseEstimator() override;

    void reset() override;
    void estimate(arma::fvec& noiseMagnitudeSpectrum, const arma::cx_fvec& signalSpectrum, std::size_t channelIndex)
        override;

    bool hasNoise();

    void setTorsoSpeed(int speed);
    int torsoSpeed() const;

    void setHeadSpeedId1(int speed);
    int headSpeedId1() const;
    void setHeadSpeedId2(int speed);
    int headSpeedId2() const;
    void setHeadSpeedId3(int speed);
    int headSpeedId3() const;
    void setHeadSpeedId4(int speed);
    int headSpeedId4() const;
    void setHeadSpeedId5(int speed);
    int headSpeedId5() const;
    void setHeadSpeedId6(int speed);
    int headSpeedId6() const;

    void setOrientationRadians(float orientationRadians);
    int orientationDegrees() const;

private:
    void verifyParameters(
        std::size_t frameSampleCount,
        std::size_t samplingFrequency,
        const std::string& noiseDirectory);
    void accumulateAPrioriNoiseMagnitudeSpectrum(
        arma::fvec& noiseMagnitudeSpectrum,
        NoiseMagnitudeDatabase& database,
        int speed,
        std::size_t channelIndex);
};

inline int round(float v, float min, float step)
{
    return static_cast<int>(step * std::round((v - min) / step) + min);
}

inline int convertSpeed(int speed, int maxSpeed)
{
    speed = std::abs(speed);
    if (speed < MinSpeed)
    {
        return 0;
    }

    speed = std::min(speed, maxSpeed);
    return round(speed, MinSpeed, SpeedStep);
}
inline int convertOrientationRadianToDegrees(float orientationRadians)
{
    orientationRadians = std::fmod(std::fmod(orientationRadians, 2 * M_PI) + 2 * M_PI, 2 * M_PI);
    int orientationDegrees = round(orientationRadians * 180.f / M_PI, MinOrientation, OrientationStep);
    return orientationDegrees % 360;
}

inline bool WeightedAverageWithAPrioriNoiseEstimator::hasNoise()
{
    return m_torsoSpeed >= MinSpeed || m_headSpeedId1 >= MinSpeed || m_headSpeedId2 >= MinSpeed ||
           m_headSpeedId3 >= MinSpeed || m_headSpeedId4 >= MinSpeed || m_headSpeedId5 >= MinSpeed ||
           m_headSpeedId6 >= MinSpeed;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setTorsoSpeed(int speed)
{
    m_torsoSpeed = convertSpeed(speed, MaxTorsoSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::torsoSpeed() const
{
    return m_torsoSpeed;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId1(int speed)
{
    m_headSpeedId1 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId1() const
{
    return m_headSpeedId1;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId2(int speed)
{
    m_headSpeedId2 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId2() const
{
    return m_headSpeedId2;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId3(int speed)
{
    m_headSpeedId3 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId3() const
{
    return m_headSpeedId3;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId4(int speed)
{
    m_headSpeedId4 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId4() const
{
    return m_headSpeedId4;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId5(int speed)
{
    m_headSpeedId5 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId5() const
{
    return m_headSpeedId5;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setHeadSpeedId6(int speed)
{
    m_headSpeedId6 = convertSpeed(speed, MaxHeadSpeed);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::headSpeedId6() const
{
    return m_headSpeedId6;
}

inline void WeightedAverageWithAPrioriNoiseEstimator::setOrientationRadians(float orientationRadians)
{
    m_orientation = convertOrientationRadianToDegrees(orientationRadians);
}

inline int WeightedAverageWithAPrioriNoiseEstimator::orientationDegrees() const
{
    return m_orientation;
}

#endif
