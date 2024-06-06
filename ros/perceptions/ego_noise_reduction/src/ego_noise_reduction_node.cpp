#include <ego_noise_reduction/WeightedAverageWithAPrioriNoiseEstimator.h>
#include <ego_noise_reduction/StftNoiseRemover.h>
#include <ego_noise_reduction/SpectralSubtractionNoiseRemover.h>
#include <ego_noise_reduction/LogMmseNoiseRemover.h>

#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>
#include <MusicBeatDetector/Utils/Exception/InvalidValueException.h>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <daemon_ros_client/msg/motor_status.hpp>

#include <hbba_lite/filters/FilterState.h>

#include <audio_utils_msgs/msg/audio_frame.hpp>

#include <armadillo>

#include <memory>
#include <queue>
#include <vector>

using namespace introlab;
using namespace std;

constexpr uint32_t AudioQueueSize = 100;
constexpr uint32_t StatusQueueSize = 1;
constexpr size_t HeadServoCount = 6;

constexpr const char* NODE_NAME = "ego_noise_reduction_node";

struct EgoNoiseReductionNodeConfiguration
{
    string typeString;
    StftNoiseRemover::Type type;
    string formatString;
    PcmAudioFrameFormat format = PcmAudioFrameFormat::Signed16;
    int channelCount;
    int samplingFrequency;
    int frameSampleCount;
    int nFft;

    string noiseDirectory;

    float noiseEstimatorEpsilon;
    float noiseEstimatorAlpha;
    float noiseEstimatorDelta;

    float spectralSubstractionAlpha0;
    float spectralSubstractionGamma;
    float spectralSubstractionBeta;

    float logMmseAlpha;
    float logMmseMaxAPosterioriSnr;
    float logMmseMinAPrioriSnr;

    EgoNoiseReductionNodeConfiguration(rclcpp::Node* node)
        : format(PcmAudioFrameFormat::Signed8),
          channelCount(0),
          samplingFrequency(0),
          frameSampleCount(0),
          nFft(0),
          noiseEstimatorEpsilon(0.f),
          noiseEstimatorAlpha(0.f),
          noiseEstimatorDelta(0.f),
          spectralSubstractionAlpha0(0.f),
          spectralSubstractionGamma(0.f),
          spectralSubstractionBeta(0.f),
          logMmseAlpha(0.f),
          logMmseMaxAPosterioriSnr(0.f),
          logMmseMinAPrioriSnr(0.f)
    {
        typeString = node->declare_parameter("type", "log_mmse");
        type = StftNoiseRemover::parseType(typeString);

        formatString = node->declare_parameter("type", "signed_32");
        format = parseFormat(formatString);

        channelCount = node->declare_parameter("channel_count", 16);
        samplingFrequency = node->declare_parameter("sampling_frequency", 16000);
        frameSampleCount = node->declare_parameter("frame_sample_count", 1024);
        nFft = node->declare_parameter("n_fft", 1024);

        noiseDirectory = node->declare_parameter("noise_directory", "");

        noiseEstimatorEpsilon = node->declare_parameter("noise_estimator_epsilon", 4.f);
        noiseEstimatorAlpha = node->declare_parameter("noise_estimator_alpha", 0.9f);
        noiseEstimatorDelta = node->declare_parameter("noise_estimator_delta", 0.9f);

        spectralSubstractionAlpha0 = node->declare_parameter("spectral_subtraction_alpha0", 5.f);
        spectralSubstractionGamma = node->declare_parameter("spectral_subtraction_gamma", 0.1f);
        spectralSubstractionBeta = node->declare_parameter("spectral_subtraction_beta", 0.01f);

        logMmseAlpha = node->declare_parameter("log_mmse_alpha", 0.98f);
        logMmseMaxAPosterioriSnr = node->declare_parameter("log_mmse_max_a_posteriori_snr", 40.f);
        logMmseMinAPrioriSnr = node->declare_parameter("log_mmse_min_a_priori_snr", 0.003f);
    }
};

class EgoNoiseReductionNode : public rclcpp::Node
{
    EgoNoiseReductionNodeConfiguration m_configuration;

    OnOffHbbaFilterState m_filterState;
    rclcpp::Publisher<audio_utils_msgs::msg::AudioFrame>::SharedPtr m_audioPub;
    rclcpp::Subscription<audio_utils_msgs::msg::AudioFrame>::SharedPtr m_audioSub;

    rclcpp::Subscription<daemon_ros_client::msg::MotorStatus>::SharedPtr m_motorStatusSub;

    PcmAudioFrame m_inputPcmAudioFrame;
    size_t m_inputPcmAudioFrameIndex;
    PackedAudioFrame<float> m_inputAudioFrame;
    PcmAudioFrame m_outputPcmAudioFrame;

    std::queue<rclcpp::Time> m_timestampQueue;
    audio_utils_msgs::msg::AudioFrame m_audioFrameMsg;

    shared_ptr<WeightedAverageWithAPrioriNoiseEstimator> m_noiseEstimator;
    unique_ptr<StftNoiseRemover> m_noiseRemover;

public:
    EgoNoiseReductionNode()
        : rclcpp::Node(NODE_NAME),
          m_configuration(this),
          m_filterState(shared_from_this(), "ego_noise_reduction/filter_state"),
          m_inputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft),
          m_inputPcmAudioFrameIndex(0),
          m_inputAudioFrame(m_configuration.channelCount, m_configuration.nFft),
          m_outputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft)
    {
        m_audioPub = create_publisher<audio_utils_msgs::msg::AudioFrame>("audio_out", AudioQueueSize);
        m_audioSub = create_subscription<audio_utils_msgs::msg::AudioFrame>(
            "audio_in",
            AudioQueueSize,
            [this](const audio_utils_msgs::msg::AudioFrame::SharedPtr msg) { audioCallback(msg); });

        m_motorStatusSub = create_subscription<daemon_ros_client::msg::MotorStatus>(
            "daemon/motor_status",
            StatusQueueSize,
            [this](const daemon_ros_client::msg::MotorStatus::SharedPtr msg) { motorStatusCallback(msg); });

        m_audioFrameMsg.format = m_configuration.formatString;
        m_audioFrameMsg.channel_count = m_configuration.channelCount;
        m_audioFrameMsg.sampling_frequency = m_configuration.samplingFrequency;
        m_audioFrameMsg.frame_sample_count = m_configuration.frameSampleCount;
        m_audioFrameMsg.data.resize(
            size(m_configuration.format, m_configuration.channelCount, m_configuration.frameSampleCount));

        m_noiseEstimator = std::make_shared<WeightedAverageWithAPrioriNoiseEstimator>(
            m_configuration.channelCount,
            m_configuration.nFft,
            m_configuration.samplingFrequency,
            m_configuration.noiseEstimatorEpsilon,
            m_configuration.noiseEstimatorAlpha,
            m_configuration.noiseEstimatorDelta,
            m_configuration.noiseDirectory);
        m_noiseRemover = createNoiseRemover();
    }

    void run() { rclcpp::spin(shared_from_this()); }

private:
    void audioCallback(const audio_utils_msgs::msg::AudioFrame::SharedPtr msg)
    {
        if (msg->format != m_configuration.formatString || msg->channel_count != m_configuration.channelCount ||
            msg->sampling_frequency != m_configuration.samplingFrequency ||
            msg->frame_sample_count != m_configuration.frameSampleCount ||
            msg->data.size() != size(m_configuration.format, msg->channel_count, msg->frame_sample_count))
        {
            RCLCPP_ERROR(
                get_logger(),
                "Not supported audio frame (msg->format=%s, msg->channel_count=%d,"
                "sampling_frequency=%d, frame_sample_count=%d, data_size=%lu)",
                msg->format.c_str(),
                msg->channel_count,
                msg->sampling_frequency,
                msg->frame_sample_count,
                msg->data.size());
            return;
        }

        m_timestampQueue.push(msg->header.stamp);
        memcpy(m_inputPcmAudioFrame.data() + m_inputPcmAudioFrameIndex, msg->data.data(), msg->data.size());
        m_inputPcmAudioFrameIndex += msg->data.size();

        if (m_inputPcmAudioFrameIndex >= m_inputPcmAudioFrame.size())
        {
            if (m_filterState.isFilteringAllMessages() || !m_noiseEstimator->hasNoise())
            {
                m_noiseEstimator->reset();
                m_noiseRemover->replaceLastFrame(m_inputPcmAudioFrame);
                publishFrames(m_inputPcmAudioFrame);
            }
            else
            {
                m_inputPcmAudioFrame.copyTo(m_inputAudioFrame);
                m_outputPcmAudioFrame = m_noiseRemover->removeNoise(m_inputAudioFrame);
                publishFrames(m_outputPcmAudioFrame);
            }
            m_inputPcmAudioFrameIndex = 0;
        }
    }

    void publishFrames(const PcmAudioFrame& frame)
    {
        for (size_t i = 0; i < frame.size(); i += m_audioFrameMsg.data.size())
        {
            m_audioFrameMsg.header.stamp = m_timestampQueue.front();
            memcpy(m_audioFrameMsg.data.data(), frame.data() + i, m_audioFrameMsg.data.size());
            m_audioPub->publish(m_audioFrameMsg);

            m_timestampQueue.pop();
        }
    }

    void motorStatusCallback(const daemon_ros_client::msg::MotorStatus::SharedPtr msg)
    {
        if (m_filterState.isFilteringAllMessages())
        {
            return;
        }

        m_noiseEstimator->setOrientationRadians(msg->torso_orientation);
        m_noiseEstimator->setTorsoSpeed(msg->torso_servo_speed);
        m_noiseEstimator->setHeadSpeedId1(msg->head_servo_speeds[0]);
        m_noiseEstimator->setHeadSpeedId2(msg->head_servo_speeds[1]);
        m_noiseEstimator->setHeadSpeedId3(msg->head_servo_speeds[2]);
        m_noiseEstimator->setHeadSpeedId4(msg->head_servo_speeds[3]);
        m_noiseEstimator->setHeadSpeedId5(msg->head_servo_speeds[4]);
        m_noiseEstimator->setHeadSpeedId5(msg->head_servo_speeds[5]);
    }

    unique_ptr<StftNoiseRemover> createNoiseRemover()
    {
        switch (m_configuration.type)
        {
            case StftNoiseRemover::Type::SpectralSubtraction:
                return std::make_unique<SpectralSubtractionNoiseRemover>(
                    m_configuration.channelCount,
                    m_configuration.nFft,
                    m_noiseEstimator,
                    m_configuration.spectralSubstractionAlpha0,
                    m_configuration.spectralSubstractionGamma,
                    m_configuration.spectralSubstractionBeta);
            case StftNoiseRemover::Type::LogMmse:
                return std::make_unique<LogMmseNoiseRemover>(
                    m_configuration.channelCount,
                    m_configuration.nFft,
                    m_noiseEstimator,
                    m_configuration.logMmseAlpha,
                    m_configuration.logMmseMaxAPosterioriSnr,
                    m_configuration.logMmseMinAPrioriSnr);
            default:
                THROW_INVALID_VALUE_EXCEPTION("type", "");
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<EgoNoiseReductionNode>();
        node->run();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger(NODE_NAME), e.what());
        return -1;
    }

    rclcpp::shutdown();

    return 0;
}
