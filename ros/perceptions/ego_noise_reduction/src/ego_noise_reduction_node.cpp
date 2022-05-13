#include <MusicBeatDetector/Utils/Data/PcmAudioFrame.h>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>

#include <hbba_lite/filters/FilterState.h>

#include <audio_utils/AudioFrame.h>

#include <armadillo>

using namespace introlab;
using namespace std;

constexpr uint32_t AUDIO_QUEUE_SIZE = 100;
constexpr uint32_t STATUS_QUEUE_SIZE = 1;

struct EgoNoiseReductionNodeConfiguration
{
    string formatString;
    PcmAudioFrameFormat format = PcmAudioFrameFormat::Signed16;
    int channelCount;
    int samplingFrequency;
    int frameSampleCount;
    int nFft;

    EgoNoiseReductionNodeConfiguration()
        : format(PcmAudioFrameFormat::Signed8),
          channelCount(0),
          samplingFrequency(0),
          frameSampleCount(0),
          nFft(0)
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

    ros::Subscriber m_currentTorsoOrientationSub;
    ros::Subscriber m_currentTorsoServoSpeedSub;
    ros::Subscriber m_currentHeadServoSpeedsSub;

    PcmAudioFrame m_inputPcmAudioFrame;
    size_t m_inputPcmAudioFrameIndex;
    PcmAudioFrame m_outputPcmAudioFrame;

    audio_utils::AudioFrame m_audioFrameMsg;

public:
    EgoNoiseReductionNode(ros::NodeHandle& nodeHandle, EgoNoiseReductionNodeConfiguration configuration)
        : m_nodeHandle(nodeHandle),
          m_configuration(move(configuration)),
          m_filterState(m_nodeHandle, "ego_noise_reduction/filter_state"),
          m_inputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft),
          m_inputPcmAudioFrameIndex(0),
          m_outputPcmAudioFrame(m_configuration.format, m_configuration.channelCount, m_configuration.nFft)
    {
        m_audioPub = m_nodeHandle.advertise<audio_utils::AudioFrame>("audio_out", AUDIO_QUEUE_SIZE);
        m_audioSub = m_nodeHandle.subscribe("audio_in", AUDIO_QUEUE_SIZE, &EgoNoiseReductionNode::audioCallback, this);

        m_currentTorsoOrientationSub = m_nodeHandle.subscribe(
            "opencr/current_torso_orientation",
            STATUS_QUEUE_SIZE,
            &EgoNoiseReductionNode::currentTorsoOrientationCallback,
            this);
        m_currentTorsoServoSpeedSub = m_nodeHandle.subscribe(
            "opencr/current_torso_servo_speed",
            STATUS_QUEUE_SIZE,
            &EgoNoiseReductionNode::currentTorsoServoSpeedCallback,
            this);
        m_currentHeadServoSpeedsSub = m_nodeHandle.subscribe(
            "opencr/current_head_servo_speeds",
            STATUS_QUEUE_SIZE,
            &EgoNoiseReductionNode::currentHeadServoSpeedsCallback,
            this);

        m_audioFrameMsg.format = m_configuration.formatString;
        m_audioFrameMsg.channel_count = m_configuration.channelCount;
        m_audioFrameMsg.sampling_frequency = m_configuration.samplingFrequency;
        m_audioFrameMsg.frame_sample_count = m_configuration.frameSampleCount;
        m_audioFrameMsg.data.resize(
            size(m_configuration.format, m_configuration.channelCount, m_configuration.frameSampleCount));
    }

    void run() { ros::spin(); }

private:
    void audioCallback(const audio_utils::AudioFramePtr& msg)
    {
        if (msg->format != m_configuration.formatString || msg->channel_count != m_configuration.channelCount ||
            msg->sampling_frequency != m_configuration.samplingFrequency ||
            msg->frame_sample_count != m_configuration.frameSampleCount)
        {
            ROS_ERROR(
                "Not supported audio frame (msg->format=%s, msg->channel_count=%d,"
                "sampling_frequency=%d, frame_sample_count=%d)",
                msg->format.c_str(),
                msg->channel_count,
                msg->sampling_frequency,
                msg->frame_sample_count);
            return;
        }

        if (m_filterState.isFilteringAllMessages())
        {
            if (m_inputPcmAudioFrameIndex != 0)
            {
                publishFrames(m_outputPcmAudioFrame, m_inputPcmAudioFrameIndex);
                m_inputPcmAudioFrameIndex = 0;
            }
            m_audioPub.publish(msg);
            return;
        }

        memcpy(m_inputPcmAudioFrame.data() + m_inputPcmAudioFrameIndex, msg->data.data(), msg->data.size());
        m_inputPcmAudioFrameIndex += msg->data.size();

        if (m_inputPcmAudioFrameIndex >= m_inputPcmAudioFrame.size())
        {
            removeNoise();
            publishFrames(m_outputPcmAudioFrame, m_outputPcmAudioFrame.size());
            m_inputPcmAudioFrameIndex = 0;
        }
    }

    void removeNoise()
    {
        // TODO
    }

    void publishFrames(const PcmAudioFrame& frame, size_t size)
    {
        for (size_t i = 0; i < size; i += m_audioFrameMsg.data.size())
        {
            memcpy(m_audioFrameMsg.data.data(), frame.data() + i, m_audioFrameMsg.data.size());
            m_audioPub.publish(m_audioFrameMsg);
        }
    }

    void currentTorsoOrientationCallback(const std_msgs::Float32Ptr& msg)
    {
        if (m_filterState.isFilteringAllMessages())
        {
            return;
        }

        // TODO
    }

    void currentTorsoServoSpeedCallback(const std_msgs::Int32Ptr& msg)
    {
        if (m_filterState.isFilteringAllMessages())
        {
            return;
        }

        // TODO
    }

    void currentHeadServoSpeedsCallback(const std_msgs::Int32MultiArrayPtr& msg)
    {
        if (m_filterState.isFilteringAllMessages())
        {
            return;
        }

        // TODO
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

        EgoNoiseReductionNode node(nodeHandle, configuration);
        node.run();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR(e.what());
        return -1;
    }

    return 0;
}
