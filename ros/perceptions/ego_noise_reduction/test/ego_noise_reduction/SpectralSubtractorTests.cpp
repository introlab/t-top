#include <ego_noise_reduction/SpectralSubtractor.h>

#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>
#include <MusicBeatDetector/Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

#include <vector>

using namespace introlab;
using namespace std;

constexpr float ALPHA_0 = 4.f;
constexpr float GAMMA = 0.1f;
constexpr float BETA = 0.01f;

string getResourcesPath()
{
    string currentFilePath = __FILE__;
    return currentFilePath.substr(0, currentFilePath.rfind("/")) + "/resources";
}

vector<PcmAudioFrame> getPcmAudioFrames(const string& path, PcmAudioFrameFormat format, size_t channelCount, size_t frameSampleCount)
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

TEST(SpectralSubtractorTests, hann_channelCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(0, 1, 4.f, 0.1f, 0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, hann_frameSampleCount0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(1, 0, 4.f, 0.1f, 0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, hann_oddFrameSampleCount_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(1, 1, 4.f, 0.1f, 0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, hann_negativeAlpha0_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(1, 1, -0.1f, 0.1f, 0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, hann_negativeGamma_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(1, 1, 4.f, -0.1f, 0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, hann_negativeBeta_shouldThrowNotSupportedException)
{
    EXPECT_THROW(SpectralSubtractor(1, 1, 4.f, 0.1f, -0.01f), NotSupportedException);
}

TEST(SpectralSubtractorTests, removeNoise_invalidChannelCount_shouldThrowNotSupportedException)
{
    constexpr size_t CHANNEL_COUNT = 2;
    constexpr size_t FRAME_SAMPLE_COUNT = 2048;
    SpectralSubtractor testee(CHANNEL_COUNT, FRAME_SAMPLE_COUNT, ALPHA_0, GAMMA, BETA);

    AudioFrame<float> frame(CHANNEL_COUNT - 1, FRAME_SAMPLE_COUNT);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FRAME_SAMPLE_COUNT / 2 + 1, CHANNEL_COUNT);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum), NotSupportedException);
}

TEST(SpectralSubtractorTests, removeNoise_invalidFrameSampleCount_shouldThrowNotSupportedException)
{
    constexpr size_t CHANNEL_COUNT = 2;
    constexpr size_t FRAME_SAMPLE_COUNT = 2048;
    SpectralSubtractor testee(CHANNEL_COUNT, FRAME_SAMPLE_COUNT, ALPHA_0, GAMMA, BETA);

    AudioFrame<float> frame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT - 1);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FRAME_SAMPLE_COUNT / 2 + 1, CHANNEL_COUNT);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum), NotSupportedException);
}

TEST(SpectralSubtractorTests, removeNoise_invalidNoiseMagnitudeSpectrum_shouldThrowNotSupportedException)
{
    constexpr size_t CHANNEL_COUNT = 2;
    constexpr size_t FRAME_SAMPLE_COUNT = 2048;
    SpectralSubtractor testee(CHANNEL_COUNT, FRAME_SAMPLE_COUNT, ALPHA_0, GAMMA, BETA);

    AudioFrame<float> frame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT);
    arma::fmat noiseMagnitudeSpectrum1 = arma::zeros<arma::fmat>(FRAME_SAMPLE_COUNT / 2, CHANNEL_COUNT);
    arma::fmat noiseMagnitudeSpectrum2 = arma::zeros<arma::fmat>(FRAME_SAMPLE_COUNT / 2 + 1, CHANNEL_COUNT - 1);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum1), NotSupportedException);
    EXPECT_THROW(testee.removeNoise(frame, noiseMagnitudeSpectrum2), NotSupportedException);
}

// TODO Finish the tests
TEST(SpectralSubtractorTests, removeNoise_zeroNoiseMagnitudeSpectrum_shouldReturnTheSameSignal)
{
    constexpr size_t CHANNEL_COUNT = 2;
    constexpr size_t FRAME_SAMPLE_COUNT = 2048;
    constexpr PcmAudioFrameFormat FORMAT = PcmAudioFrameFormat::Signed32;

    SpectralSubtractor testee(CHANNEL_COUNT, FRAME_SAMPLE_COUNT, ALPHA_0, GAMMA, BETA);

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", FORMAT, CHANNEL_COUNT, FRAME_SAMPLE_COUNT);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds_zero_output.raw", FORMAT, CHANNEL_COUNT, FRAME_SAMPLE_COUNT);

    AudioFrame<float> inputFrame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT);
    arma::fmat noiseMagnitudeSpectrum = arma::zeros<arma::fmat>(FRAME_SAMPLE_COUNT / 2 + 1, CHANNEL_COUNT);
    AudioFrame<float> expectedOutputFrame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT);

    const size_t frameCount = min(inputPcmFrames.size(), expectedOutputPcmFrames.size());
    for (size_t i = 0; i < frameCount; i++)
    {
        inputPcmFrames[i].copyTo(inputFrame);
        AudioFrame<float> outputFrame = testee.removeNoise(inputFrame, noiseMagnitudeSpectrum);
        expectedOutputPcmFrames[i].copyTo(expectedOutputFrame);

        expectFrameNear(outputFrame, expectedOutputFrame, 0.001);
    }
}

TEST(SpectralSubtractorTests, removeNoise_shouldRemoveTheNoise)
{
    constexpr size_t CHANNEL_COUNT = 2;
    constexpr size_t FRAME_SAMPLE_COUNT = 2048;
    constexpr PcmAudioFrameFormat FORMAT = PcmAudioFrameFormat::Signed32;

    SpectralSubtractor testee(CHANNEL_COUNT, FRAME_SAMPLE_COUNT, ALPHA_0, GAMMA, BETA);

    string resourcesPath = getResourcesPath();
    vector<PcmAudioFrame> inputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds.raw", FORMAT, CHANNEL_COUNT, FRAME_SAMPLE_COUNT);
    vector<PcmAudioFrame> expectedOutputPcmFrames = getPcmAudioFrames(resourcesPath + "/noisy_sounds_output.raw", FORMAT, CHANNEL_COUNT, FRAME_SAMPLE_COUNT);

    AudioFrame<float> inputFrame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT);
    arma::fmat noiseMagnitudeSpectrum;
    noiseMagnitudeSpectrum.load(resourcesPath + "/noises.txt");
    AudioFrame<float> expectedOutputFrame(CHANNEL_COUNT, FRAME_SAMPLE_COUNT);

    const size_t frameCount = min(inputPcmFrames.size(), expectedOutputPcmFrames.size());
    for (size_t i = 0; i < frameCount; i++)
    {
        inputPcmFrames[i].copyTo(inputFrame);
        AudioFrame<float> outputFrame = testee.removeNoise(inputFrame, noiseMagnitudeSpectrum);
        expectedOutputPcmFrames[i].copyTo(expectedOutputFrame);

        expectFrameNear(outputFrame, expectedOutputFrame, 0.001);
    }
}
