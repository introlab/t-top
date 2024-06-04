#include <rclcpp/rclcpp.hpp>
#include <rosbag2_generic_topic/rosbag2_node.hpp>

#include <sys/types.h>
#include <signal.h>

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace std;
namespace fs = std::filesystem;

string readFile(const fs::path& path)
{
    std::ifstream t(path);
    std::stringstream ss;
    ss << t.rdbuf();
    return ss.str();
}

bool getNodePid(const string& nodeName, int& pid)
{
    auto arg = "__node:=" + nodeName;

    for (const auto& entry : fs::directory_iterator("/proc"))
    {
        if (readFile(entry.path() / "cmdline").find(arg) != string::npos)
        {
            pid = stoi(entry.path().filename());
            return true;
        }
    }

    return false;
}

class WatchdogNode : public rosbag2_generic_topic::Rosbag2Node
{
    string m_nodeName;
    string m_topic;
    chrono::milliseconds m_timeoutDuration;

    rclcpp::TimerBase::SharedPtr m_initTimer;

    std::shared_ptr<rosbag2_generic_topic::GenericSubscription> m_subscriber;
    rclcpp::TimerBase::SharedPtr m_messageTimer;

public:
    WatchdogNode() : rosbag2_generic_topic::Rosbag2Node("watchdog_node")
    {
        m_nodeName = declare_parameter("node_name", "");
        m_topic = rclcpp::expand_topic_or_service_name(declare_parameter("topic", ""), get_name(), get_namespace(), false);
        RCLCPP_INFO_STREAM(get_logger(), "Listening on: " << m_topic);

        double timeoutDurationS = declare_parameter("timeout_duration_s", 60.0);
        m_timeoutDuration = chrono::milliseconds(static_cast<int>(timeoutDurationS * 1000));

        m_initTimer = create_wall_timer(std::chrono::seconds(1), [this]() { initTimerCallback(); });
    }

    void run() { rclcpp::spin(shared_from_this()); }

private:
    void initTimerCallback()
    {
        auto all_topics_and_types = get_topic_names_and_types();
        auto it = all_topics_and_types.find(m_topic);
        if (it == all_topics_and_types.end() || it->second.empty())
        {
            return;
        }

        RCLCPP_INFO_STREAM(get_logger(), "Topic type found: " << it->second[0]);

        m_subscriber = create_generic_subscription(m_topic, it->second[0], 1,
            [this](std::shared_ptr<rclcpp::SerializedMessage> msg)
            {
                topicCallback();
            });

        m_initTimer->cancel();
    }

    void topicCallback()
    {
        if (m_messageTimer)
        {
            m_messageTimer->reset();
        }
        else
        {
            m_messageTimer = create_wall_timer(m_timeoutDuration, [this]() { messageTimerCallback(); });
        }
    }

    void messageTimerCallback()
    {
        RCLCPP_ERROR_STREAM(get_logger(), "The node '" << m_nodeName << "' is no longer responding. The node will be respawned.");
        killNode(m_nodeName);

        m_messageTimer->cancel();
        m_messageTimer = nullptr;
    }

    void killNode(const string& nodeName)
    {
        int pid;
        if (getNodePid(nodeName, pid))
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Killing PID: " << pid);
            kill(pid, SIGINT);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<WatchdogNode>();
    node->run();

    rclcpp::shutdown();

    return 0;
}
