#include <ego_noise_reduction/WeightedAverageWithAPrioriNoiseEstimator.h>

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <fstream>

using namespace introlab;
using namespace std;

WeightedAverageWithAPrioriNoiseEstimator::WeightedAverageWithAPrioriNoiseEstimator(
    size_t channelCount,
    size_t frameSampleCount,
    size_t samplingFrequency,
    float epsilon,
    float alpha,
    float delta,
    const string& noiseDirectory)
    : NoiseEstimator(channelCount, frameSampleCount),
      m_noiseEstimator(channelCount, frameSampleCount, epsilon, alpha, delta),
      m_magnitudeSize(frameSampleCount / 2 + 1),
      m_torsoDatabase(
          noiseDirectory,
          "torso_servo",
          m_magnitudeSize,
          MinSpeed,
          MaxTorsoSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId1(
          noiseDirectory,
          "head_servo_id1",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId2(
          noiseDirectory,
          "head_servo_id2",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId3(
          noiseDirectory,
          "head_servo_id3",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId4(
          noiseDirectory,
          "head_servo_id4",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId5(
          noiseDirectory,
          "head_servo_id5",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_headDatabaseId6(
          noiseDirectory,
          "head_servo_id6",
          m_magnitudeSize,
          MinSpeed,
          MaxHeadSpeed,
          SpeedStep,
          MinOrientation,
          MaxOrientation,
          OrientationStep,
          channelCount),
      m_torsoSpeed(0),
      m_headSpeedId1(0),
      m_headSpeedId2(0),
      m_headSpeedId3(0),
      m_headSpeedId4(0),
      m_headSpeedId5(0),
      m_headSpeedId6(0),
      m_orientation(0)
{
    verifyParameters(frameSampleCount, samplingFrequency, noiseDirectory);

    m_estimatedNoiseMagnitudeSpectrum.zeros(m_magnitudeSize);
    m_aPrioriNoiseMagnitudeSpectrum.zeros(m_magnitudeSize);
}

WeightedAverageWithAPrioriNoiseEstimator::~WeightedAverageWithAPrioriNoiseEstimator() {}

void WeightedAverageWithAPrioriNoiseEstimator::reset()
{
    m_noiseEstimator.reset();
}

void WeightedAverageWithAPrioriNoiseEstimator::estimate(
    arma::fvec& noiseMagnitudeSpectrum,
    const arma::cx_fvec& signalSpectrum,
    size_t channelIndex)
{
    m_noiseEstimator.estimate(m_estimatedNoiseMagnitudeSpectrum, signalSpectrum, channelIndex);

    noiseMagnitudeSpectrum.zeros(m_magnitudeSize);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_torsoDatabase, m_torsoSpeed, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId1, m_headSpeedId1, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId2, m_headSpeedId2, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId3, m_headSpeedId3, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId4, m_headSpeedId4, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId5, m_headSpeedId5, channelIndex);
    accumulateAPrioriNoiseMagnitudeSpectrum(noiseMagnitudeSpectrum, m_headDatabaseId6, m_headSpeedId6, channelIndex);

    noiseMagnitudeSpectrum = arma::max(noiseMagnitudeSpectrum, m_estimatedNoiseMagnitudeSpectrum);
}

void WeightedAverageWithAPrioriNoiseEstimator::verifyParameters(
    size_t frameSampleCount,
    size_t samplingFrequency,
    const string& noiseDirectory)
{
    ifstream file(noiseDirectory + "/info.txt");
    if (!file.good())
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The noise directory does not contain info.txt.");
    }

    string line;
    getline(file, line);
    if (stoi(line) != frameSampleCount)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Invalid noise directory N_FFT.");
    }

    getline(file, line);
    if (stoi(line) != samplingFrequency)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Invalid noise directory sampling frequency.");
    }

    getline(file, line);
    if (stoi(line) != channelCount())
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Invalid noise directory channel count.");
    }
}

void WeightedAverageWithAPrioriNoiseEstimator::accumulateAPrioriNoiseMagnitudeSpectrum(
    arma::fvec& noiseMagnitudeSpectrum,
    NoiseMagnitudeDatabase& database,
    int speed,
    size_t channelIndex)
{
    if (speed >= MinSpeed)
    {
        database.noiseMagnitude(m_aPrioriNoiseMagnitudeSpectrum, speed, m_orientation, channelIndex);
        noiseMagnitudeSpectrum += m_aPrioriNoiseMagnitudeSpectrum;
    }
}
