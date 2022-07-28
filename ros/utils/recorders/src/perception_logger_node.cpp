#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>
#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <video_analyzer/VideoAnalysis.h>
#include <audio_analyzer/AudioAnalysis.h>

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

    unique_ptr<VideoAnalysisLogger> m_videoAnalysisLogger;
    unique_ptr<AudioAnalysisLogger> m_audioAnalysisLogger;

    tf::TransformListener m_listener;

    ros::Subscriber m_videoAnalysis3dSubscriber;
    ros::Subscriber m_videoAnalysis2dWideSubscriber;
    ros::Subscriber m_audioAnalysisSubscriber;

public:
    PerceptionLoggerNode(ros::NodeHandle& nodeHandle, PerceptionLoggerNodeConfiguration configuration)
        : m_nodeHandle(nodeHandle),
          m_configuration(move(configuration))
    {
        SQLite::Database database(m_configuration.databasePath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

        m_videoAnalysisLogger = make_unique<SQLiteVideoAnalysisLogger>(database);
        m_audioAnalysisLogger = make_unique<SQLiteAudioAnalysisLogger>(database);

        m_videoAnalysis3dSubscriber =
            m_nodeHandle.subscribe("video_analysis", 10, &PerceptionLoggerNode::videoAnalysisSubscriberCallback, this);
        m_audioAnalysisSubscriber =
            m_nodeHandle.subscribe("audio_analysis", 10, &PerceptionLoggerNode::audioAnalysisSubscriberCallback, this);
        ;
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
            ROS_ERROR(ex.what());
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

        m_audioAnalysisLogger->log(msgToAnalysis(msg));
    }

    void run() { ros::spin(); }

private:
    Position pointToPosition(const geometry_msgs::Point& point, const tf::StampedTransform& transform)
    {
        tf::Point transformedPoint = transform * tf::Point(point.x, point.y, point.z);
        return Position(transformedPoint.x(), transformedPoint.y(), transformedPoint.z());
    }

    Direction positionToDirection(const Position& p)
    {
        double n = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        return Direction(p.x / n, p.y / n, p.z / n);
    }

    VideoAnalysis msgToAnalysis(
        const video_analyzer::VideoAnalysisObject& msg,
        const ros::Time& timestamp,
        const tf::StampedTransform& transform)
    {
        Position position = pointToPosition(msg.center_3d, transform);
        VideoAnalysis videoAnalysis(timestamp, position, positionToDirection(position), msg.object_class);

        if (!msg.person_pose_3d.empty())
        {
            videoAnalysis.personPose = vector<Position>();
            videoAnalysis.personPose.value().reserve(msg.person_pose_3d.size());
            for (auto& point : msg.person_pose_3d)
            {
                videoAnalysis.personPose.value().push_back(pointToPosition(point, transform));
            }
        }
        if (!msg.person_pose_confidence.empty())
        {
            videoAnalysis.personPoseConfidence = msg.person_pose_confidence;
        }
        if (!msg.face_descriptor.empty())
        {
            videoAnalysis.faceDescriptor = msg.face_descriptor;
        }
    }

    AudioAnalysis msgToAnalysis(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
    {
        AudioAnalysis audioAnalysis(
            msg->header.stamp,
            Direction(msg->direction_x, msg->direction_y, msg->direction_z),
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
