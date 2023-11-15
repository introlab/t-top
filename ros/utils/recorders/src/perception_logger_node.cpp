#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>
#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>
#include <perception_logger/sqlite/SQLiteSpeechLogger.h>
#include <perception_logger/sqlite/SQLiteHbbaStrategyStateLogger.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <video_analyzer/VideoAnalysis.h>
#include <audio_analyzer/AudioAnalysis.h>
#include <talk/Text.h>
#include <speech_to_text/Transcript.h>
#include <hbba_lite/StrategyState.h>

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

class PerceptionLoggerNode
{
    ros::NodeHandle& m_nodeHandle;
    PerceptionLoggerNodeConfiguration m_configuration;

    SQLite::Database m_database;
    unique_ptr<VideoAnalysisLogger> m_videoAnalysisLogger;
    unique_ptr<AudioAnalysisLogger> m_audioAnalysisLogger;
    unique_ptr<SpeechLogger> m_speechLogger;
    unique_ptr<HbbaStrategyStateLogger> m_hbbaStrategyStateLogger;

    tf::TransformListener m_listener;

    ros::Subscriber m_videoAnalysis3dSubscriber;
    ros::Subscriber m_audioAnalysisSubscriber;
    ros::Subscriber m_talkTextSubscriber;
    ros::Subscriber m_speechToTextTranscriptSubscriber;
    ros::Subscriber m_hbbaStrategyStateSubscriber;

public:
    PerceptionLoggerNode(ros::NodeHandle& nodeHandle, PerceptionLoggerNodeConfiguration configuration)
        : m_nodeHandle(nodeHandle),
          m_configuration(move(configuration)),
          m_database(m_configuration.databasePath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE)
    {
        m_videoAnalysisLogger = make_unique<SQLiteVideoAnalysisLogger>(m_database);
        m_audioAnalysisLogger = make_unique<SQLiteAudioAnalysisLogger>(m_database);
        m_speechLogger = make_unique<SQLiteSpeechLogger>(m_database);
        m_hbbaStrategyStateLogger = make_unique<SQLiteHbbaStrategyStateLogger>(m_database);

        m_videoAnalysis3dSubscriber =
            m_nodeHandle.subscribe("video_analysis", 10, &PerceptionLoggerNode::videoAnalysisSubscriberCallback, this);
        m_audioAnalysisSubscriber =
            m_nodeHandle.subscribe("audio_analysis", 10, &PerceptionLoggerNode::audioAnalysisSubscriberCallback, this);
        m_talkTextSubscriber =
            m_nodeHandle.subscribe("talk/text", 10, &PerceptionLoggerNode::talkTextSubscriberCallback, this);
        m_speechToTextTranscriptSubscriber = m_nodeHandle.subscribe(
            "speech_to_text/transcript",
            10,
            &PerceptionLoggerNode::speechToTextTranscriptSubscriberCallback,
            this);
        m_hbbaStrategyStateSubscriber = m_nodeHandle.subscribe(
            "hbba_strategy_state_log",
            10,
            &PerceptionLoggerNode::hbbaStrategyStateSubscriberCallback,
            this);
    }

    virtual ~PerceptionLoggerNode() {}

    void videoAnalysisSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg)
    {
        if (!msg->contains_3d_positions)
        {
            ROS_ERROR("The video analysis must contain 3d positions.");
            return;
        }

        tf::StampedTransform transform;
        try
        {
            m_listener.lookupTransform(m_configuration.frameId, msg->header.frame_id, msg->header.stamp, transform);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            return;
        }

        for (auto& object : msg->objects)
        {
            m_videoAnalysisLogger->log(msgToAnalysis(object, msg->header.stamp, transform));
        }
    }

    void audioAnalysisSubscriberCallback(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
    {
        if (msg->header.frame_id != m_configuration.frameId)
        {
            ROS_ERROR_STREAM(
                "Invalid direction frame id (" << msg->header.frame_id << " ! = " << m_configuration.frameId);
            return;
        }
        if (msg->audio_classes.empty())
        {
            return;
        }

        m_audioAnalysisLogger->log(msgToAnalysis(msg));
    }

    void talkTextSubscriberCallback(const talk::Text::ConstPtr& msg)
    {
        m_speechLogger->log(Speech(ros::Time::now(), SpeechSource::ROBOT, msg->text));
    }

    void speechToTextTranscriptSubscriberCallback(const speech_to_text::Transcript::ConstPtr& msg)
    {
        if (msg->is_final)
        {
            m_speechLogger->log(Speech(ros::Time::now(), SpeechSource::HUMAN, msg->text));
        }
    }

    void hbbaStrategyStateSubscriberCallback(const hbba_lite::StrategyState::ConstPtr& msg)
    {
        m_hbbaStrategyStateLogger->log(
            HbbaStrategyState(ros::Time::now(), msg->desire_type_name, msg->strategy_type_name, msg->enabled));
    }

    void run() { ros::spin(); }

private:
    static Position pointToPosition(const geometry_msgs::Point& point, const tf::StampedTransform& transform)
    {
        tf::Point transformedPoint = transform * tf::Point(point.x, point.y, point.z);
        return Position{transformedPoint.x(), transformedPoint.y(), transformedPoint.z()};
    }

    static ImagePosition pointToImagePosition(const geometry_msgs::Point& point)
    {
        return ImagePosition{point.x, point.y};
    }

    static BoundingBox centreWidthHeightToBoundingBox(const geometry_msgs::Point& centre, double width, double height)
    {
        return BoundingBox{pointToImagePosition(centre), width, height};
    }

    static Direction positionToDirection(const Position& p)
    {
        double n = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        return Direction{p.x / n, p.y / n, p.z / n};
    }

    static VideoAnalysis msgToAnalysis(
        const video_analyzer::VideoAnalysisObject& msg,
        const ros::Time& timestamp,
        const tf::StampedTransform& transform)
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
                [&transform](const geometry_msgs::Point& point) { return pointToPosition(point, transform); });
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

    static AudioAnalysis msgToAnalysis(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
    {
        AudioAnalysis audioAnalysis(
            msg->header.stamp,
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
    ros::init(argc, argv, "perception_logger_node");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    PerceptionLoggerNodeConfiguration configuration;

    if (!privateNodeHandle.getParam("database_path", configuration.databasePath))
    {
        ROS_ERROR("The parameter database_path is required.");
        return -1;
    }
    if (!privateNodeHandle.getParam("frame_id", configuration.frameId))
    {
        ROS_ERROR("The parameter frame_id is required.");
        return -1;
    }

    PerceptionLoggerNode node(nodeHandle, configuration);
    node.run();

    return 0;
}
