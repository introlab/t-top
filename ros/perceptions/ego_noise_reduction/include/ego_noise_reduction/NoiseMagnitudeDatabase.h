#ifndef EGO_NOISE_REDUCTION_NOISE_MAGNITUDE_DATABASE_H
#define EGO_NOISE_REDUCTION_NOISE_MAGNITUDE_DATABASE_H

#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <armadillo>

#include <string>
#include <unordered_map>
#include <fstream>

class NoiseMagnitudeDatabase
{
    std::unordered_map<int, arma::fvec> m_noiseMagnitudesBySpeed;
    std::unordered_map<int, arma::fvec> m_tfByOrientation;
    std::unordered_map<std::size_t, arma::fvec> m_tfByChannelIndex;

public:
    NoiseMagnitudeDatabase(
        const std::string& directory,
        const std::string& prefix,
        std::size_t magnitudeSize,
        int minSpeed,
        int maxSpeed,
        int speedStep,
        int minOrientation,
        int maxOrientation,
        int orientationStep,
        std::size_t channelCount);

    void noiseMagnitude(arma::fvec& noiseMagnitude, int speed, int orientation, std::size_t channelIndex);

private:
    template<class T>
    void loadFile(std::unordered_map<T, arma::fvec>& output, const std::string& path);

    void verifyKeys(
        int minSpeed,
        int maxSpeed,
        int speedStep,
        int minOorientation,
        int maxOrientation,
        int orientationStep,
        std::size_t channelCount);

    template<class T>
    void verifyValues(const std::unordered_map<T, arma::fvec>& map, std::size_t magnitudeSize);
};

template<class T>
void NoiseMagnitudeDatabase::loadFile(std::unordered_map<T, arma::fvec>& output, const std::string& path)
{
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line))
    {
        size_t separatorIndex = line.find('|');
        std::string key = line.substr(0, separatorIndex);
        std::string value = line.substr(separatorIndex + 1);
        output[static_cast<T>(std::stoi(key))] = arma::fvec(value);
    }
}

template<class T>
void NoiseMagnitudeDatabase::verifyValues(const std::unordered_map<T, arma::fvec>& map, std::size_t magnitudeSize)
{
    for (auto pair : map)
    {
        if (pair.second.n_elem != magnitudeSize)
        {
            THROW_NOT_SUPPORTED_EXCEPTION(
                "A vector does not have the right size (" + std::to_string(pair.second.n_elem) +
                "!= " + std::to_string(magnitudeSize) + ")");
        }
    }
}

#endif
