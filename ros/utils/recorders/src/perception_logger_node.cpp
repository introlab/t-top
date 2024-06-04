#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>
#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>
#include <perception_logger/sqlite/SQLiteSpeechLogger.h>
#include <perception_logger/sqlite/SQLiteHbbaStrategyStateLogger.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <video_analyzer/msg/video_analysis.hpp>
#include <audio_analyzer/msg/audio_analysis.hpp>
#include <talk/msg/text.hpp>
#include <speech_to_text/msg/transcript.hpp>
#include <hbba_lite/msg/strategy_state.hpp>

#include <memory>
#include <sstream>
#include <cmath>

using namespace std;

string mergeClasses(const vector<string>& classes)
{
    std::ostringstream mergedClasses;
    for (size_t i = 0; i < classes.size();)
    {
        mergedClasses << classes[i];
        i++;

        if (i < classes.size())
        {
            mergedClasses << ",";
        }
    }

    return mergedClasses.str();
}

struct PerceptionLoggerNodeConfiguration
{
    string databasePath;
    string frameId;
};

class PerceptionLoggerNode : public rclcpp::Node
{
    string m_databasePath;
    string m_frameId;

    unique_ptr<SQLite::Database> m_database;
    unique_ptr<VideoAnalysisLogger> m_videoAnalysisLogger;
    unique_ptr<AudioAnalysisLogger> m_audioAnalysisLogger;
    unique_ptr<SpeechLogger> m_speechLogger;
    unique_ptr<HbbaStrategyStateLogger> m_hbbaStrategyStateLogger;

    unique_ptr<tf2_ros::Buffer> m_tfBuffer;
    shared_ptr<tf2_ros::TransformListener> m_tfListener;

    rclcpp::Subscription<video_analyzer::msg::VideoAnalysis>::SharedPtr m_videoAnalysis3dSubscriber;
    rclcpp::Subscription<audio_analyzer::msg::AudioAnalysis>::SharedPtr m_audioAnalysisSubscriber;
    rclcpp::Subscription<talk::msg::Text>::SharedPtr m_talkTextSubscriber;
    rclcpp::Subscription<speech_to_text::msg::Transcript>::SharedPtr m_speechToTextTranscriptSubscriber;
    rclcpp::Subscription<hbba_lite::msg::StrategyState>::SharedPtr m_hbbaStrategyStateSubscriber;

public:
    PerceptionLoggerNode() : rclcpp::Node("perception_logger_node")
    {
        m_databasePath = declare_parameter("database_path", "");
        m_frameId = declare_parameter("frame_id", "");

        m_database = std::make_unique<SQLite::Database>(m_databasePath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

        m_videoAnalysisLogger = std::make_unique<SQLiteVideoAnalysisLogger>(*m_database);
        m_audioAnalysisLogger = std::make_unique<SQLiteAudioAnalysisLogger>(*m_database);
        m_speechLogger = std::make_unique<SQLiteSpeechLogger>(*m_database);
        m_hbbaStrategyStateLogger = std::make_unique<SQLiteHbbaStrategyStateLogger>(*m_database);

        m_videoAnalysis3dSubscriber = create_subscription<video_analyzer::msg::VideoAnalysis>(
            "video_analysis",
            10,
            [this](const video_analyzer::msg::VideoAnalysis::SharedPtr msg) { videoAnalysisSubscriberCallback(msg); });
        m_audioAnalysisSubscriber = create_subscription<audio_analyzer::msg::AudioAnalysis>(
            "audio_analysis",
            10,
            [this](const audio_analyzer::msg::AudioAnalysis::SharedPtr msg) { audioAnalysisSubscriberCallback(msg); });
        m_talkTextSubscriber = create_subscription<talk::msg::Text>(
            "talk/text",
            10,
            [this](const talk::msg::Text::SharedPtr msg) { talkTextSubscriberCallback(msg); });
        m_speechToTextTranscriptSubscriber = create_subscription<speech_to_text::msg::Transcript>(
            "speech_to_text/transcript",
            10,
            [this](const speech_to_text::msg::Transcript::SharedPtr msg)
            { speechToTextTranscriptSubscriberCallback(msg); });
        m_hbbaStrategyStateSubscriber = create_subscription<hbba_lite::msg::StrategyState>(
            "hbba_strategy_state_log",
            10,
            [this](const hbba_lite::msg::StrategyState::SharedPtr msg) { hbbaStrategyStateSubscriberCallback(msg); });

        m_tfBuffer = std::make_unique<tf2_ros::Buffer>(get_clock());
        m_tfListener = std::make_shared<tf2_ros::TransformListener>(*m_tfBuffer);
    }

    virtual ~PerceptionLoggerNode() {}

    void videoAnalysisSubscriberCallback(const video_analyzer::msg::VideoAnalysis::SharedPtr msg)
    {
        if (!msg->contains_3d_positions)
        {
            RCLCPP_ERROR(get_logger(), "The video analysis must contain 3d positions.");
            return;
        }

        geometry_msgs::msg::TransformStamped transformMsg;
        try
        {
            transformMsg = m_tfBuffer->lookupTransform(m_frameId, msg->header.frame_id, msg->header.stamp);
        }
        catch (tf2::TransformException& ex)
        {
            RCLCPP_ERROR(get_logger(), "%s", ex.what());
            return;
        }
        tf2::Stamped<tf2::Transform> transform;
        tf2::convert(transformMsg, transform);

        for (auto& object : msg->objects)
        {
            m_videoAnalysisLogger->log(msgToAnalysis(object, msg->header.stamp, transform));
        }
    }

    void audioAnalysisSubscriberCallback(const audio_analyzer::msg::AudioAnalysis::SharedPtr msg)
    {
        if (msg->header.frame_id != m_frameId)
        {
            RCLCPP_ERROR_STREAM(
                get_logger(),
                "Invalid direction frame id (" << msg->header.frame_id << " ! = " << m_frameId);
            return;
        }
        if (msg->audio_classes.empty())
        {
            return;
        }

        m_audioAnalysisLogger->log(msgToAnalysis(msg));
    }

    void talkTextSubscriberCallback(const talk::msg::Text::SharedPtr msg)
    {
        m_speechLogger->log(Speech(get_clock()->now(), SpeechSource::ROBOT, msg->text));
    }

    void speechToTextTranscriptSubscriberCallback(const speech_to_text::msg::Transcript::SharedPtr msg)
    {
        if (msg->is_final)
        {
            m_speechLogger->log(Speech(get_clock()->now(), SpeechSource::HUMAN, msg->text));
        }
    }

    void hbbaStrategyStateSubscriberCallback(const hbba_lite::msg::StrategyState::SharedPtr msg)
    {
        m_hbbaStrategyStateLogger->log(
            HbbaStrategyState(get_clock()->now(), msg->desire_type_name, msg->strategy_type_name, msg->enabled));
    }

    void run() { rclcpp::spin(shared_from_this()); }

private:
    static Position
        pointToPosition(const geometry_msgs::msg::Point& point, const tf2::Stamped<tf2::Transform>& transform)
    {
        tf2::Vector3 transformedPoint = transform * tf2::Vector3(point.x, point.y, point.z);
        return Position{transformedPoint.x(), transformedPoint.y(), transformedPoint.z()};
    }

    static ImagePosition pointToImagePosition(const geometry_msgs::msg::Point& point)
    {
        return ImagePosition{point.x, point.y};
    }

    static BoundingBox
        centreWidthHeightToBoundingBox(const geometry_msgs::msg::Point& centre, double width, double height)
    {
        return BoundingBox{pointToImagePosition(centre), width, height};
    }

    static Direction positionToDirection(const Position& p)
    {
        double n = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        return Direction{p.x / n, p.y / n, p.z / n};
    }

    static VideoAnalysis msgToAnalysis(
        const video_analyzer::msg::VideoAnalysisObject& msg,
        const rclcpp::Time& timestamp,
        const tf2::Stamped<tf2::Transform>& transform)
    {
        Position position = pointToPosition(msg.center_3d, transform);
        VideoAnalysis videoAnalysis{
            timestamp,
            position,
            positionToDirection(position),
            msg.object_class,
            msg.object_confidence,
            msg.object_class_probability,
            centreWidthHeightToBoundingBox(msg.center_2d, msg.width_2d, msg.height_2d)};

        if (!msg.person_pose_2d.empty())
        {
            videoAnalysis.personPoseImage = vector<ImagePosition>();
            videoAnalysis.personPoseImage->reserve(msg.person_pose_2d.size());
            std::transform(
                std::begin(msg.person_pose_2d),
                std::end(msg.person_pose_2d),
                std::back_inserter(videoAnalysis.personPoseImage.value()),
                pointToImagePosition);
        }
        if (!msg.person_pose_3d.empty())
        {
            videoAnalysis.personPose = vector<Position>();
            videoAnalysis.personPose.value().reserve(msg.person_pose_3d.size());
            std::transform(
                std::begin(msg.person_pose_3d),
                std::end(msg.person_pose_3d),
                std::back_inserter(videoAnalysis.personPose.value()),
                [&transform](const geometry_msgs::msg::Point& point) { return pointToPosition(point, transform); });
        }
        if (!msg.person_pose_confidence.empty())
        {
            videoAnalysis.personPoseConfidence = msg.person_pose_confidence;
        }
        if (!msg.face_descriptor.empty())
        {
            videoAnalysis.faceAlignmentKeypointCount = msg.face_alignment_keypoint_count;
            videoAnalysis.faceSharpnessScore = msg.face_sharpness_score;
            videoAnalysis.faceDescriptor = msg.face_descriptor;
        }

        return videoAnalysis;
    }

    static AudioAnalysis msgToAnalysis(const audio_analyzer::msg::AudioAnalysis::SharedPtr msg)
    {
        AudioAnalysis audioAnalysis(
            rclcpp::Time(msg->header.stamp),
            Direction{msg->direction_x, msg->direction_y, msg->direction_z},
            msg->tracking_id,
            mergeClasses(msg->audio_classes));

        if (!msg->voice_descriptor.empty())
        {
            audioAnalysis.voiceDescriptor = msg->voice_descriptor;
        }

        return audioAnalysis;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = make_shared<PerceptionLoggerNode>();
    node->run();

    rclcpp::shutdown();

    return 0;
}
