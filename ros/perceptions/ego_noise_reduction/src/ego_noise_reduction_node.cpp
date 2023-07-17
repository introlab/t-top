#include <ego_noise_reduction/WeightedAverageWithAPrioriNoiseEstimator.h>
#include <ego_noise_reduction/StftNoiseRemover.h>
#include <ego_noise_reduction/SpectralSubtractionNoiseRemover.h>
#include <ego_noise_reduction/LogMmseNoiseRemover.h>

#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>
#include <MusicBeatDetector/Utils/Exception/InvalidValueException.h>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>

#include <daemon_ros_client/MotorStatus.h>

#include <hbba_lite/filters/FilterState.h>

#include <audio_utils/AudioFrame.h>

#include <armadillo>

#include <memory>
#include <queue>
#include <vector>

using namespace introlab;
using namespace std;

constexpr uint32_t AudioQueueSize = 100;
constexpr uint32_t StatusQueueSize = 1;
constexpr size_t HeadServoCount = 6;

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

    EgoNoiseReductionNodeConfiguration()
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
    }
};

class EgoNoiseReductionNode
{
    ros::NodeHandle& m_nodeHandle;
    EgoNoiseReductionNodeConfiguration m_configuration;

    OnOffHbbaFilterState m_filterState;
    ros::Publisher m_audioPub;
    ros::Subscriber m_audioSub;

    ros::Subscriber m_motorStatusSub;

    PcmAudioFrame m_inputPcmAudioFrame;
    size_t m_inputPcmAudioFrameIndex;
    PackedAudioFrame<float> m_inputAudioFrame;
    PcmAudioFrame m_outputPcmAudioFrame;

    std::queue<ros::Time> m_timestampQueue;
    audio_utils::AudioFrame m_audioFrameMsg;

    shared_ptr<WeightedAverageWithAPrioriNoiseEstimator> m_noiseEstimator;
    unique_ptr<StftNoiseRemover> m_noiseRemover;

public:
    EgoNoiseReductionNode(ros::NodeHandle& nodeHandle, EgoNoiseReductionNodeConfiguration configuration)
        : m_nodeHandle(nodeHandle),
          m_configuration(move(configuration)),
          m_filterState(m_nodeHandle, "ego_noise_reduction/filter_state"),
          m_inputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft),
          m_inputPcmAudioFrameIndex(0),
          m_inputAudioFrame(m_configuration.channelCount, m_configuration.nFft),
          m_outputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft)
    {
        m_audioPub = m_nodeHandle.advertise<audio_utils::AudioFrame>("audio_out", AudioQueueSize);
        m_audioSub = m_nodeHandle.subscribe("audio_in", AudioQueueSize, &EgoNoiseReductionNode::audioCallback, this);

        m_motorStatusSub = m_nodeHandle.subscribe("daemon/motor_status", StatusQueueSize, &EgoNoiseReductionNode::motorStatusCallback, this);

        m_audioFrameMsg.format = m_configuration.formatString;
        m_audioFrameMsg.channel_count = m_configuration.channelCount;
        m_audioFrameMsg.sampling_frequency = m_configuration.samplingFrequency;
        m_audioFrameMsg.frame_sample_count = m_configuration.frameSampleCount;
        m_audioFrameMsg.data.resize(
            size(m_configuration.format, m_configuration.channelCount, m_configuration.frameSampleCount));

        m_noiseEstimator = make_shared<WeightedAverageWithAPrioriNoiseEstimator>(
            m_configuration.channelCount,
            m_configuration.nFft,
            m_configuration.samplingFrequency,
            m_configuration.noiseEstimatorEpsilon,
            m_configuration.noiseEstimatorAlpha,
            m_configuration.noiseEstimatorDelta,
            m_configuration.noiseDirectory);
        m_noiseRemover = createNoiseRemover();
    }

    void run() { ros::spin(); }

private:
    void audioCallback(const audio_utils::AudioFrame::ConstPtr& msg)
    {
        if (msg->format != m_configuration.formatString || msg->channel_count != m_configuration.channelCount ||
            msg->sampling_frequency != m_configuration.samplingFrequency ||
            msg->frame_sample_count != m_configuration.frameSampleCount ||
            msg->data.size() != size(m_configuration.format, msg->channel_count, msg->frame_sample_count))
        {
            ROS_ERROR(
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
            m_audioPub.publish(m_audioFrameMsg);

            m_timestampQueue.pop();
        }
    }

    void motorStatusCallback(const daemon_ros_client::MotorStatus::ConstPtr& msg)
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
                return make_unique<SpectralSubtractionNoiseRemover>(
                    m_configuration.channelCount,
                    m_configuration.nFft,
                    m_noiseEstimator,
                    m_configuration.spectralSubstractionAlpha0,
                    m_configuration.spectralSubstractionGamma,
                    m_configuration.spectralSubstractionBeta);
            case StftNoiseRemover::Type::LogMmse:
                return make_unique<LogMmseNoiseRemover>(
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
    ros::init(argc, argv, "ego_noise_reduction_node");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    EgoNoiseReductionNodeConfiguration configuration;

    try
    {
        if (!privateNodeHandle.getParam("type", configuration.typeString))
        {
            ROS_ERROR("The parameter type must be spectral_subtraction or log_mmse.");
            return -1;
        }
        configuration.type = StftNoiseRemover::parseType(configuration.typeString);

        if (!privateNodeHandle.getParam("format", configuration.formatString))
        {
            ROS_ERROR("The parameter format is required.");
            return -1;
        }
        configuration.format = parseFormat(configuration.formatString);

        if (!privateNodeHandle.getParam("channel_count", configuration.channelCount))
        {
            ROS_ERROR("The parameter channel_count is required.");
            return -1;
        }
        if (!privateNodeHandle.getParam("sampling_frequency", configuration.samplingFrequency))
        {
            ROS_ERROR("The parameter sampling_frequency is required.");
            return -1;
        }
        if (!privateNodeHandle.getParam("frame_sample_count", configuration.frameSampleCount))
        {
            ROS_ERROR("The parameter frame_sample_count is required.");
            return -1;
        }
        if (!privateNodeHandle.getParam("n_fft", configuration.nFft) ||
            (configuration.nFft % configuration.frameSampleCount) != 0)
        {
            ROS_ERROR("The parameter n_fft is required. It must be a multiple of frame_sample_count.");
            return -1;
        }
        if (!privateNodeHandle.getParam("noise_directory", configuration.noiseDirectory))
        {
            ROS_ERROR("The parameter noise_directory is required.");
            return -1;
        }

        configuration.noiseEstimatorEpsilon = privateNodeHandle.param("noise_estimator_epsilon", 4.f);
        configuration.noiseEstimatorAlpha = privateNodeHandle.param("noise_estimator_alpha", 0.9f);
        configuration.noiseEstimatorDelta = privateNodeHandle.param("noise_estimator_delta", 0.9f);

        configuration.spectralSubstractionAlpha0 = privateNodeHandle.param("spectral_subtraction_alpha0", 5.f);
        configuration.spectralSubstractionGamma = privateNodeHandle.param("spectral_subtraction_gamma", 0.1f);
        configuration.spectralSubstractionBeta = privateNodeHandle.param("spectral_subtraction_beta", 0.01f);

        configuration.logMmseAlpha = privateNodeHandle.param("log_mmse_alpha", 0.98f);
        configuration.logMmseMaxAPosterioriSnr = privateNodeHandle.param("log_mmse_max_a_posteriori_snr", 40.f);
        configuration.logMmseMinAPrioriSnr = privateNodeHandle.param("log_mmse_min_a_priori_snr", 0.003f);

        EgoNoiseReductionNode node(nodeHandle, configuration);
        node.run();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM(e.what());
        return -1;
    }

    return 0;
}
