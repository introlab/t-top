#ifndef _DAEMON_ROS_CLIENT_NODE_H_
#define _DAEMON_ROS_CLIENT_NODE_H_

#include "WebSocketProtocolWrapper.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <daemon_ros_client/msg/base_status.hpp>
#include <daemon_ros_client/msg/motor_status.hpp>
#include <daemon_ros_client/msg/led_color.hpp>
#include <daemon_ros_client/msg/led_colors.hpp>

#include <SerialCommunication.h>


using namespace std;

constexpr uint32_t PubQueueSize = 1;
constexpr uint32_t SubQueueSize = 10;
constexpr const char* HEAD_POSE_FRAME_ID = "stewart_base";

class DaemonRosClientNode : public QObject, public rclcpp::Node
{
    Q_OBJECT

    double m_baseLinkTorsoBaseDeltaZ;

    WebSocketProtocolWrapper *m_websocketProtocolWrapper;

    rclcpp::Publisher<daemon_ros_client::msg::BaseStatus>::SharedPtr m_baseStatusPub;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr m_startButtonPressedPub;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr m_stopButtonPressedPub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr m_imuPub;
    rclcpp::Publisher<daemon_ros_client::msg::MotorStatus>::SharedPtr m_motorStatusPub;

    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr m_setVolumeSub;
    rclcpp::Subscription<daemon_ros_client::msg::LedColors>::SharedPtr m_setLedColorsSub;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr m_setTorsoOrientationSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr m_setHeadPoseSub;

    tf2_ros::TransformBroadcaster m_tfBroadcaster;

public:
    DaemonRosClientNode();

    void cleanup();

private:
    void initROS();
    void initWebSocketProtocolWrapper();
    void setVolumeCallback(const std_msgs::msg::UInt8::SharedPtr& msg);
    void setLedColorsCallback(const daemon_ros_client::msg::LedColors::SharedPtr& msg);
    void setTorsoOrientationCallback(const std_msgs::msg::Float32::SharedPtr& msg);
    void setHeadPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr& msg);
    void handleBaseStatus(Device source, const BaseStatusPayload& payload);
    void handleButtonPressed(Device source, const ButtonPressedPayload& payload);
    void handleImuData(Device source, const ImuDataPayload& payload);
    void handleMotorStatus(Device source, const MotorStatusPayload& payload);
    void sendTorsoTf(const rclcpp::Time& stamp, float torsoOrientation);
    void sendHeadTf(const rclcpp::Time& stamp, const geometry_msgs::msg::Pose& pose);
};

#endif //_DAEMON_ROS_CLIENT_NODE_H_
