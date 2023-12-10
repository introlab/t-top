#ifndef _DAEMON_ROS_CLIENT_NODE_H_
#define _DAEMON_ROS_CLIENT_NODE_H_

#include <QCoreApplication>
#include "WebSocketProtocolWrapper.h"

#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <daemon_ros_client/BaseStatus.h>
#include <daemon_ros_client/MotorStatus.h>
#include <daemon_ros_client/LedColor.h>
#include <daemon_ros_client/LedColors.h>

#include <SerialCommunication.h>


using namespace std;

constexpr uint32_t PubQueueSize = 1;
constexpr uint32_t SubQueueSize = 10;
constexpr const char* HEAD_POSE_FRAME_ID = "stewart_base";

struct DaemonRosClientNodeConfiguration
{
    double baseLinkTorsoBaseDeltaZ;

    DaemonRosClientNodeConfiguration() : baseLinkTorsoBaseDeltaZ(0.0)
    {

    }
};

class DaemonRosClientNode : public QCoreApplication
{

    WebSocketProtocolWrapper *m_websocketProtocolWrapper;

    ros::NodeHandle& m_nodeHandle;
    DaemonRosClientNodeConfiguration m_configuration;

    ros::Publisher m_baseStatusPub;
    ros::Publisher m_startButtonPressedPub;
    ros::Publisher m_stopButtonPressedPub;
    ros::Publisher m_imuPub;
    ros::Publisher m_motorStatusPub;

    ros::Subscriber m_setVolumeSub;
    ros::Subscriber m_setLedColorsSub;
    ros::Subscriber m_setTorsoOrientationSub;
    ros::Subscriber m_setHeadPoseSub;

    tf2_ros::TransformBroadcaster m_tfBroadcaster;

    Q_OBJECT

    public:

    DaemonRosClientNode(int &argc, char* argv[], ros::NodeHandle& nodeHandle, DaemonRosClientNodeConfiguration configuration);

    void cleanup();

    private:

    void initROS();
    void initWebSocketProtocolWrapper();
    void setVolumeCallback(const std_msgs::UInt8::ConstPtr& msg);
    void setLedColorsCallback(const daemon_ros_client::LedColors::ConstPtr& msg);
    void setTorsoOrientationCallback(const std_msgs::Float32::ConstPtr& msg);
    void setHeadPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void handleBaseStatus(Device source, const BaseStatusPayload& payload);
    void handleButtonPressed(Device source, const ButtonPressedPayload& payload);
    void handleImuData(Device source, const ImuDataPayload& payload);
    void handleMotorStatus(Device source, const MotorStatusPayload& payload);
    void sendTorsoTf(const ros::Time& stamp, float torsoOrientation);
    void sendHeadTf(const ros::Time& stamp, const geometry_msgs::Pose& pose);
};

#endif //_DAEMON_ROS_CLIENT_NODE_H_
