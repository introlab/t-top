#include <ros/ros.h>
#include <ros/network.h>
#include <ros/xmlrpc_manager.h>
#include <xmlrpcpp/XmlRpc.h>

#include <topic_tools/shape_shifter.h>

#include <sys/types.h>
#include <signal.h>

using namespace std;

constexpr bool ONE_SHOT = true;

bool getNodePid(const string& nodeName, int& pid)
{
    XmlRpc::XmlRpcValue req;
    req[0] = ros::this_node::getName();
    req[1] = nodeName;

    XmlRpc::XmlRpcValue resp;
    XmlRpc::XmlRpcValue payload;
    if (!ros::master::execute("lookupNode", req, resp, payload, true))
    {
        ROS_ERROR_STREAM("Failed to lookup the watched node (" << nodeName << ").");
        return false;
    }

    string peerHost;
    uint32_t peerPort;
    if (!ros::network::splitURI(static_cast<std::string>(resp[2]), peerHost, peerPort))
    {
        ROS_ERROR_STREAM("Invalid API URI (" << resp[2] << ")");
        return false;
    }

    XmlRpc::XmlRpcClient c(peerHost.c_str(), peerPort, "/");
    XmlRpc::XmlRpcValue req2;
    XmlRpc::XmlRpcValue resp2;
    req2[0] = ros::this_node::getName();
    c.execute("getPid", req2, resp2);
    if (c.isFault() || !resp2.valid() || static_cast<int>(resp2[0]) != 1)
    {
        ROS_ERROR_STREAM("Failed to fetch the pid of " << nodeName << ".");
        return false;
    }

    pid = static_cast<int>(resp2[2]);
    return true;
}

void killNode(const string& nodeName)
{
    int pid;
    if (getNodePid(nodeName, pid))
    {
        ROS_ERROR_STREAM("Killing PID: " << pid);
        kill(pid, SIGINT);
    }
}

class WatchdogNode
{
    ros::NodeHandle& m_nodeHandle;

    string m_nodeName;
    ros::Duration m_timeoutDuration;

    ros::Time m_lastStartupTime;

    ros::Subscriber m_subscriber;
    ros::Timer m_messageTimer;

public:
    WatchdogNode(ros::NodeHandle& nodeHandle, string nodeName, ros::Duration timeoutDuration)
        : m_nodeHandle(nodeHandle),
          m_nodeName(move(nodeName)),
          m_timeoutDuration(timeoutDuration)
    {
        m_lastStartupTime = ros::Time::now();

        m_subscriber = nodeHandle.subscribe("topic", 1, &WatchdogNode::topicCallback, this);
    }

    void run() { ros::spin(); }

private:
    void topicCallback(const ros::MessageEvent<topic_tools::ShapeShifter>& msgEvent)
    {
        if (m_messageTimer.isValid())
        {
            m_messageTimer.stop();
        }

        m_messageTimer =
            m_nodeHandle.createTimer(m_timeoutDuration, &WatchdogNode::messageTimerCallback, this, ONE_SHOT);
    }

    void messageTimerCallback(const ros::TimerEvent&)
    {
        ROS_ERROR_STREAM("The node '" << m_nodeName << "' is no longer responding. The node will be respawned.");
        killNode(m_nodeName);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "beat_detector_node");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    string nodeName;
    double timeoutDurationS;

    if (!privateNodeHandle.getParam("node_name", nodeName))
    {
        ROS_ERROR("The parameter node_name is required.");
        return -1;
    }
    if (!privateNodeHandle.getParam("timeout_duration_s", timeoutDurationS))
    {
        ROS_ERROR("The parameter timeout_duration_s is required.");
        return -1;
    }

    WatchdogNode node(nodeHandle, nodeName, ros::Duration(timeoutDurationS));
    node.run();

    return 0;
}
