#include "TestUtils.h"

#include <gtest/gtest.h>

#include <fstream>

using namespace introlab;
using namespace std;

constexpr float ABS_ERROR = 0.001f;

string getResourcesPath()
{
    string currentFilePath = __FILE__;
    return currentFilePath.substr(0, currentFilePath.rfind("/")) + "/resources";
}

vector<PcmAudioFrame>
    getPcmAudioFrames(const string& path, PcmAudioFrameFormat format, size_t channelCount, size_t frameSampleCount)
{
    std::ifstream file(path, std::ifstream::in);
    vector<PcmAudioFrame> frames;

    while (file.good())
    {
        PcmAudioFrame frame(format, channelCount, frameSampleCount);
        file >> frame;
        frames.push_back(frame);
    }

    return frames;
}

void writePcmAudioFrames(const string& path, const vector<PcmAudioFrame>& frames)
{
    std::ofstream file(path, std::ifstream::out);
    for (auto& frame : frames)
    {
        file << frame;
    }
}

void expectFrameNear(const AudioFrame<float> value, const AudioFrame<float> expected, float absError)
{
    ASSERT_EQ(value.size(), expected.size());

    for (size_t i = 0; i < value.size(); i++)
    {
        EXPECT_NEAR(value[i], expected[i], absError);
    }
}

void testNoiseReduction(
    StftNoiseRemover& remover,
    const vector<PcmAudioFrame>& inputPcmFrames,
    const vector<PcmAudioFrame>& expectedOutputPcmFrames,
    const arma::fmat& noiseMagnitudeSpectrum)
{
    ASSERT_GT(inputPcmFrames.size(), 0);

    const size_t channelCount = inputPcmFrames[0].channelCount();
    const size_t frameSampleCount = inputPcmFrames[0].channelCount();

    AudioFrame<float> inputFrame(channelCount, frameSampleCount);
    AudioFrame<float> expectedOutputFrame(channelCount, frameSampleCount);

    const size_t frameCount = min(inputPcmFrames.size(), expectedOutputPcmFrames.size());
    for (size_t i = 0; i < frameCount; i++)
    {
        inputPcmFrames[i].copyTo(inputFrame);
        AudioFrame<float> outputFrame = remover.removeNoise(inputFrame, noiseMagnitudeSpectrum);
        expectedOutputPcmFrames[i].copyTo(expectedOutputFrame);

        expectFrameNear(outputFrame, expectedOutputFrame, ABS_ERROR);
    }
}
