#include <ego_noise_reduction/NoiseMagnitudeDatabase.h>

using namespace std;

NoiseMagnitudeDatabase::NoiseMagnitudeDatabase(
    const string& directory,
    const string& prefix,
    size_t magnitudeSize,
    int minSpeed,
    int maxSpeed,
    int speedStep,
    int minOrientation,
    int maxOrientation,
    int orientationStep,
    size_t channelCount)
{
    loadFile(m_noiseMagnitudesBySpeed, directory + "/" + prefix + "_base_noise_magnitudes.txt");
    loadFile(m_tfByOrientation, directory + "/" + prefix + "_orientation_tf.txt");
    loadFile(m_tfByChannelIndex, directory + "/" + prefix + "_channel_tf.txt");

    verifyKeys(minSpeed, maxSpeed, speedStep, minOrientation, maxOrientation, orientationStep, channelCount);
    verifyValues(m_noiseMagnitudesBySpeed, magnitudeSize);
    verifyValues(m_tfByOrientation, magnitudeSize);
    verifyValues(m_tfByChannelIndex, magnitudeSize);
}

void NoiseMagnitudeDatabase::noiseMagnitude(arma::fvec& noiseMagnitude, int speed, int orientation, size_t channelIndex)
{
    noiseMagnitude =
        m_noiseMagnitudesBySpeed[speed] % m_tfByOrientation[orientation] % m_tfByChannelIndex[channelIndex];
}

void NoiseMagnitudeDatabase::verifyKeys(
    int minSpeed,
    int maxSpeed,
    int speedStep,
    int minOrientation,
    int maxOrientation,
    int orientationStep,
    size_t channelCount)
{
    for (int speed = minSpeed; speed <= maxSpeed; speed += speedStep)
    {
        if (m_noiseMagnitudesBySpeed.find(speed) == m_noiseMagnitudesBySpeed.end())
        {
            THROW_NOT_SUPPORTED_EXCEPTION("The speed " + to_string(speed) + " is missing in the database files.");
        }
    }

    for (int orientation = minOrientation; orientation <= maxOrientation; orientation += orientationStep)
    {
        if (m_tfByOrientation.find(orientation) == m_tfByOrientation.end())
        {
            THROW_NOT_SUPPORTED_EXCEPTION(
                "The orientation " + to_string(orientation) + " is missing in the database files.");
        }
    }

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    {
        if (m_tfByChannelIndex.find(channelIndex) == m_tfByChannelIndex.end())
        {
            THROW_NOT_SUPPORTED_EXCEPTION(
                "The channel " + to_string(channelIndex) + " is missing in the database files.");
        }
    }
}
