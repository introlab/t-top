#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <daemon_ros_client/msg/motor_status.hpp>

#include <unordered_set>
#include <optional>

using namespace std;

constexpr const char* NODE_NAME = "set_head_pose_arbitration_node";
constexpr const char* HEAD_POSE_FRAME_ID = "stewart_base";

struct Topic
{
    string name;
    int priority;
    rclcpp::Duration timeout;
};


geometry_msgs::msg::Pose defaultPose()
{
    geometry_msgs::msg::Pose pose;

    pose.position.x = 0.0;
    pose.position.y = 0.0;
    pose.position.z = 0.0;

    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;

    return pose;
}


class ArbitrationNode : public rclcpp::Node
{
    vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> m_subscribers;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_publisher;

    vector<Topic> m_topics;
    optional<int> m_currentTopicIndex;
    rclcpp::Time m_lastMessageTime;

    vector<geometry_msgs::msg::Pose> m_offsets;

    rclcpp::Subscription<daemon_ros_client::msg::MotorStatus>::SharedPtr m_motorStatusSubscriber;
    optional<geometry_msgs::msg::PoseStamped> m_lastPose;

public:
    ArbitrationNode()
        : rclcpp::Node(NODE_NAME),
          m_lastMessageTime(get_clock()->now())
    {
        m_topics = convertToTopics(
            declare_parameter("topics", vector<string>{}),
            declare_parameter("priorities", vector<int64_t>{}),
            declare_parameter("timeout_s", vector<double>{}));
        if (!hasUniquePriority(m_topics))
        {
            throw std::runtime_error("The topic priorities must be unique.");
        }

        for (size_t i = 0; i < m_topics.size(); i++)
        {
            m_subscribers.emplace_back(create_subscription<geometry_msgs::msg::PoseStamped>(
                m_topics[i].name,
                1,
                [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { callback(i, msg); }));
        }

        auto offsetTopics = declare_parameter("offset_topics", vector<string>{});
        for (size_t i = 0; i < offsetTopics.size(); i++)
        {
            m_subscribers.emplace_back(create_subscription<geometry_msgs::msg::PoseStamped>(
                offsetTopics[i],
                1,
                [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { offsetCallback(i, msg); }));
            m_offsets.emplace_back(defaultPose());
        }

        m_publisher = create_publisher<geometry_msgs::msg::PoseStamped>("daemon/set_head_pose", 1);

        m_motorStatusSubscriber =
            create_subscription<daemon_ros_client::msg::MotorStatus>("daemon/motor_status", 1, [this](const daemon_ros_client::msg::MotorStatus::SharedPtr msg) { motorStatusCallback(msg); });
    }

    void run() { rclcpp::spin(shared_from_this()); }

private:
    vector<Topic> convertToTopics(const vector<string>& topics, const vector<int64_t>& priorities, const vector<double>& timeoutS)
    {
        if (topics.size() != priorities.size() || topics.size() != timeoutS.size())
        {
            throw std::runtime_error("The topics, priorities, timeout_s parameters must have the same size.");
        }

        vector<Topic> convertedTopics;
        for (size_t i = 0; i < topics.size(); i++)
        {
            auto expandedTopic = rclcpp::expand_topic_or_service_name(topics[i], get_name(), get_namespace(), false);
            auto timeout = chrono::duration<double, std::ratio<1>>(timeoutS[i]);
            convertedTopics.emplace_back(Topic{expandedTopic, static_cast<int>(priorities[i]), timeout});
        }

        return convertedTopics;
    }

    bool hasUniquePriority(const vector<Topic>& topics)
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

    void callback(size_t i, const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Invalid head pose frame id (" << msg->header.frame_id << ")");
            return;
        }

        if (m_currentTopicIndex == nullopt || m_topics[i].priority <= m_topics[*m_currentTopicIndex].priority ||
            (get_clock()->now() - m_lastMessageTime) > m_topics[*m_currentTopicIndex].timeout)
        {
            m_currentTopicIndex = i;
            m_lastMessageTime = get_clock()->now();
            m_publisher->publish(applyOffsets(*msg));
            m_lastPose = *msg;
        }
    }

    void offsetCallback(size_t i, const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Invalid head pose frame id (" << msg->header.frame_id << ")");
            return;
        }

        m_offsets[i] = msg->pose;
        if (m_lastPose.has_value())
        {
            m_publisher->publish(applyOffsets(*m_lastPose));
        }
    }

    geometry_msgs::msg::PoseStamped applyOffsets(const geometry_msgs::msg::PoseStamped& inputMsg)
    {
        double x = inputMsg.pose.position.x;
        double y = inputMsg.pose.position.y;
        double z = inputMsg.pose.position.z;

        tf2::Quaternion rotation(
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
                tf2::Quaternion(offset.orientation.x, offset.orientation.y, offset.orientation.z, offset.orientation.w);
        }

        geometry_msgs::msg::PoseStamped outputMsg;
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

    void motorStatusCallback(const daemon_ros_client::msg::MotorStatus::SharedPtr msg)
    {
        if (m_lastPose == nullopt)
        {
            m_lastPose = geometry_msgs::msg::PoseStamped();
            m_lastPose->header.frame_id = msg->head_pose_frame_id;
            m_lastPose->pose = msg->head_pose;
        }
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<ArbitrationNode>();
        node->run();
        rclcpp::shutdown();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger(NODE_NAME), "%s", e.what());
        rclcpp::shutdown();
        return -1;
    }

    return 0;
}
