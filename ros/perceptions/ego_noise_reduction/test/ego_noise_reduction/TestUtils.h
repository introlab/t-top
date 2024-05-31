#ifndef EGO_NOISE_REDUCTION_TEST_UTILS_H
#define EGO_NOISE_REDUCTION_TEST_UTILS_H

#include <ego_noise_reduction/NoiseEstimator.h>
#include <ego_noise_reduction/StftNoiseRemover.h>

#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>

#include <vector>
#include <string>

std::string getResourcesPath();

std::vector<introlab::PcmAudioFrame> getPcmAudioFrames(
    const std::string& path,
    introlab::PcmAudioFrameFormat format,
    size_t channelCount,
    size_t frameSampleCount);

void writePcmAudioFrames(const std::string& path, const std::vector<introlab::PcmAudioFrame>& frames);

void expectFrameNear(
    const introlab::PackedAudioFrame<float> value,
    const introlab::PackedAudioFrame<float> expected,
    float absError);

void testNoiseReduction(
    StftNoiseRemover& remover,
    const std::vector<introlab::PcmAudioFrame>& inputPcmFrames,
    const std::vector<introlab::PcmAudioFrame>& expectedOutputPcmFrames);

class ConstantNoiseEstimator : public NoiseEstimator
{
    arma::fmat m_noiseMagnitudeSpectrum;

public:
    ConstantNoiseEstimator(arma::fmat noiseMagnitudeSpectrum);
    ~ConstantNoiseEstimator() override;

    void reset() override;
    void
        estimate(arma::fvec& noiseMagnitudeSpectrum, const arma::cx_fvec& signalSpectrum, size_t channelIndex) override;
};

std::shared_ptr<NoiseEstimator>
    createZeroConstantNoiseEstimator(std::size_t channelCount, std::size_t frameSampleCount);
std::shared_ptr<NoiseEstimator> createConstantNoiseEstimatorFromFile(const std::string& path);

#endif
