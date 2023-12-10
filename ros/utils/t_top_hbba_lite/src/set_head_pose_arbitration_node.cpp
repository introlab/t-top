#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>

#include <daemon_ros_client/MotorStatus.h>

#include <unordered_set>
#include <optional>

using namespace std;

constexpr const char* HEAD_POSE_FRAME_ID = "stewart_base";

struct Topic
{
    string name;
    int priority;
    ros::Duration timeout;
};


geometry_msgs::Pose defaultPose()
{
    geometry_msgs::Pose pose;

    pose.position.x = 0.0;
    pose.position.y = 0.0;
    pose.position.z = 0.0;

    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;

    return pose;
}


class ArbitrationNode
{
    ros::NodeHandle& m_nodeHandle;
    vector<ros::Subscriber> m_subscribers;
    ros::Publisher m_publisher;
    bool m_hasAdvertised;

    vector<Topic> m_topics;
    optional<int> m_currentTopicIndex;
    ros::Time m_lastMessageTime;

    vector<geometry_msgs::Pose> m_offsets;

    ros::Subscriber m_motorStatusSubscriber;
    optional<geometry_msgs::PoseStamped> m_lastPose;

public:
    ArbitrationNode(ros::NodeHandle& nodeHandle, vector<Topic> topics, vector<string> offsetTopics, bool latch)
        : m_nodeHandle(nodeHandle),
          m_hasAdvertised(false),
          m_topics(move(topics)),
          m_lastMessageTime(ros::Time::now())
    {
        for (size_t i = 0; i < m_topics.size(); i++)
        {
            m_subscribers.emplace_back(m_nodeHandle.subscribe<geometry_msgs::PoseStamped>(
                m_topics[i].name,
                1,
                [this, i](const geometry_msgs::PoseStamped::ConstPtr& msg) { callback(i, msg); }));
        }

        for (size_t i = 0; i < offsetTopics.size(); i++)
        {
            m_subscribers.emplace_back(m_nodeHandle.subscribe<geometry_msgs::PoseStamped>(
                offsetTopics[i],
                1,
                [this, i](const geometry_msgs::PoseStamped::ConstPtr& msg) { offsetCallback(i, msg); }));
            m_offsets.emplace_back(defaultPose());
        }

        m_publisher = m_nodeHandle.advertise<geometry_msgs::PoseStamped>("daemon/set_head_pose", 1, latch);

        m_motorStatusSubscriber =
            m_nodeHandle.subscribe("daemon/motor_status", 1, &ArbitrationNode::motorStatusCallback, this);
    }

    void run() { ros::spin(); }

private:
    void callback(size_t i, const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
        {
            ROS_ERROR_STREAM("Invalid head pose frame id (" << msg->header.frame_id << ")");
            return;
        }

        if (m_currentTopicIndex == nullopt || m_topics[i].priority <= m_topics[*m_currentTopicIndex].priority ||
            (ros::Time::now() - m_lastMessageTime) > m_topics[*m_currentTopicIndex].timeout)
        {
            m_currentTopicIndex = i;
            m_lastMessageTime = ros::Time::now();
            m_publisher.publish(applyOffsets(*msg));
            m_lastPose = *msg;
        }
    }

    void offsetCallback(size_t i, const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
        {
            ROS_ERROR_STREAM("Invalid head pose frame id (" << msg->header.frame_id << ")");
            return;
        }

        m_offsets[i] = msg->pose;
        if (m_lastPose.has_value())
        {
            m_publisher.publish(applyOffsets(*m_lastPose));
        }
    }

    geometry_msgs::PoseStamped applyOffsets(const geometry_msgs::PoseStamped& inputMsg)
    {
        double x = inputMsg.pose.position.x;
        double y = inputMsg.pose.position.y;
        double z = inputMsg.pose.position.z;

        tf::Quaternion rotation(
            inputMsg.pose.orientation.x,
            inputMsg.pose.orientation.y,
            inputMsg.pose.orientation.z,
            inputMsg.pose.orientation.w);

        for (auto& offset : m_offsets)
        {
            x += offset.position.x;
            y += offset.position.y;
            z += offset.position.z;

            rotation *=
                tf::Quaternion(offset.orientation.x, offset.orientation.y, offset.orientation.z, offset.orientation.w);
        }

        geometry_msgs::PoseStamped outputMsg;
        outputMsg.header = inputMsg.header;
        outputMsg.pose.position.x = x;
        outputMsg.pose.position.y = y;
        outputMsg.pose.position.z = z;
        outputMsg.pose.orientation.x = rotation.getX();
        outputMsg.pose.orientation.y = rotation.getY();
        outputMsg.pose.orientation.z = rotation.getZ();
        outputMsg.pose.orientation.w = rotation.getW();
        return outputMsg;
    }

    void motorStatusCallback(const daemon_ros_client::MotorStatus::ConstPtr& msg)
    {
        if (m_lastPose == nullopt)
        {
            m_lastPose = geometry_msgs::PoseStamped();
            m_lastPose->header.frame_id = msg->head_pose_frame_id;
            m_lastPose->pose = msg->head_pose;
        }
    }
};


bool getTopics(ros::NodeHandle& privateNodeHandle, vector<Topic>& topics)
{
    vector<string> value;

    XmlRpc::XmlRpcValue xmlTopics;
    privateNodeHandle.getParam("topics", xmlTopics);
    if (xmlTopics.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        ROS_ERROR("Invalid topics format");
        return false;
    }

    for (size_t i = 0; i < xmlTopics.size(); i++)
    {
        if (xmlTopics[i].getType() != XmlRpc::XmlRpcValue::TypeStruct ||
            xmlTopics[i]["name"].getType() != XmlRpc::XmlRpcValue::TypeString ||
            xmlTopics[i]["priority"].getType() != XmlRpc::XmlRpcValue::TypeInt ||
            xmlTopics[i]["timeout_s"].getType() != XmlRpc::XmlRpcValue::TypeDouble)
        {
            ROS_ERROR_STREAM("Invalid topics[" << i << "]: name must be a string and priority must be a int and.");
            return false;
        }

        topics.emplace_back(Topic{
            static_cast<string>(xmlTopics[i]["name"]),
            static_cast<int>(xmlTopics[i]["priority"]),
            ros::Duration(static_cast<double>(xmlTopics[i]["timeout_s"]))});
    }

    return topics.size() > 0;
}

bool hasUniquePriorities(const vector<Topic>& topics)
{
    unordered_set<int> priorities;

    for (auto& topic : topics)
    {
        if (priorities.count(topic.priority) > 0)
        {
            return false;
        }

        priorities.insert(topic.priority);
    }

    return true;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "arbitration_node");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    vector<Topic> topics;
    if (!getTopics(privateNodeHandle, topics))
    {
        ROS_ERROR("The parameter topics must be set, not empty and valid.");
        return -1;
    }
    if (!hasUniquePriorities(topics))
    {
        ROS_ERROR("The topic priorities must be unique.");
        return -1;
    }

    vector<string> offsetTopics;
    if (!privateNodeHandle.getParam("offset_topics", offsetTopics))
    {
        ROS_ERROR("The parameter offset_topics must be set.");
        return false;
    }

    bool latch;
    if (!privateNodeHandle.getParam("latch", latch))
    {
        ROS_ERROR("The parameter latch is required.");
        return -1;
    }

    ArbitrationNode node(nodeHandle, topics, offsetTopics, latch);
    node.run();

    return 0;
}
