#ifndef _QT_WEBSOCKET_APP_H_
#define _QT_WEBSOCKET_APP_H_

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
    DaemonRosClientNode(int &argc, char* argv[], ros::NodeHandle& nodeHandle, DaemonRosClientNodeConfiguration configuration)
        : QCoreApplication(argc, argv), m_websocketProtocolWrapper(nullptr), m_nodeHandle(nodeHandle), m_configuration(move(configuration))
    {
        initWebSocketProtocolWrapper();
        initROS();

    }

    void initROS()
    {
        m_baseStatusPub = m_nodeHandle.advertise<daemon_ros_client::BaseStatus>("daemon/base_status", PubQueueSize);
        m_startButtonPressedPub = m_nodeHandle.advertise<std_msgs::Empty>("daemon/start_button_pressed", PubQueueSize);
        m_stopButtonPressedPub = m_nodeHandle.advertise<std_msgs::Empty>("daemon/stop_button_pressed", PubQueueSize);
        m_imuPub = m_nodeHandle.advertise<sensor_msgs::Imu>("daemon/imu/data_raw", PubQueueSize);
        m_motorStatusPub = m_nodeHandle.advertise<daemon_ros_client::MotorStatus>("daemon/motor_status", PubQueueSize);

        m_setVolumeSub = m_nodeHandle.subscribe("daemon/set_volume", SubQueueSize, &DaemonRosClientNode::setVolumeCallback, this);
        m_setLedColorsSub = m_nodeHandle.subscribe("daemon/set_led_colors", SubQueueSize, &DaemonRosClientNode::setLedColorsCallback, this);
        m_setTorsoOrientationSub = m_nodeHandle.subscribe("daemon/set_torso_orientation", SubQueueSize, &DaemonRosClientNode::setTorsoOrientationCallback, this);
        m_setHeadPoseSub = m_nodeHandle.subscribe("daemon/set_head_pose", SubQueueSize, &DaemonRosClientNode::setHeadPoseCallback, this);
    }

    void initWebSocketProtocolWrapper()
    {
        QUrl url(WebSocketProtocolWrapper::ROS_DEFAULT_CLIENT_URL);
        m_websocketProtocolWrapper = new WebSocketProtocolWrapper(url, this);
    }

    private:

    void setVolumeCallback(const std_msgs::UInt8::ConstPtr& msg)
    {
        SetVolumePayload payload;
        payload.volume = msg->data;

        // TODO send to Device::PSU_CONTROL
    }

    void setLedColorsCallback(const daemon_ros_client::LedColors::ConstPtr& msg)
    {
        SetLedColorsPayload payload;
        for (size_t i  = 0; i < SetLedColorsPayload::LED_COUNT; i++)
        {
            payload.colors[i].red = msg->colors[i].red;
            payload.colors[i].green = msg->colors[i].green;
            payload.colors[i].blue = msg->colors[i].blue;
        }

        // TODO send to Device::PSU_CONTROL
    }

    void setTorsoOrientationCallback(const std_msgs::Float32::ConstPtr& msg)
    {
        SetVolumePayload payload;
        payload.volume = msg->data;

        // TODO send to Device::DYNAMIXEL_CONTROL
    }

    void setHeadPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
        {
            ROS_ERROR_STREAM("Invalid head pose frame id (" << msg->header.frame_id << ")");
            return;
        }

        SetHeadPosePayload payload;
        payload.headPosePositionX = msg->pose.position.x;
        payload.headPosePositionY = msg->pose.position.y;
        payload.headPosePositionZ = msg->pose.position.z;
        payload.headPoseOrientationW = msg->pose.orientation.w;
        payload.headPoseOrientationX = msg->pose.orientation.x;
        payload.headPoseOrientationY = msg->pose.orientation.y;
        payload.headPoseOrientationZ = msg->pose.orientation.z;

        // TODO send to Device::DYNAMIXEL_CONTROL
    }

    void handleBaseStatus(Device source, const BaseStatusPayload& payload)
    {
        daemon_ros_client::BaseStatus msg;
        msg.header.stamp = ros::Time::now();
        msg.is_psu_connected = payload.isPsuConnected;
        msg.has_charger_error = payload.hasChargerError;
        msg.is_battery_charging = payload.isBatteryCharging;
        msg.has_battery_error = payload.hasBatteryError;
        msg.state_of_charge = payload.stateOfCharge;
        msg.current = payload.current;
        msg.voltage = payload.voltage;
        msg.onboard_temperature = payload.onboardTemperature;
        msg.external_temperature = payload.externalTemperature;
        msg.front_light_sensor = payload.frontLightSensor;
        msg.back_light_sensor = payload.backLightSensor;
        msg.left_light_sensor = payload.leftLightSensor;
        msg.right_light_sensor = payload.rightLightSensor;
        msg.volume = payload.volume;
        msg.maximum_volume = payload.maximumVolume;

        m_baseStatusPub.publish(msg);
    }

    void handleButtonPressed(Device source, const ButtonPressedPayload& payload)
    {
        switch (payload.button)
        {
            case Button::START:
                m_startButtonPressedPub.publish(std_msgs::Empty());
                break;
            case Button::STOP:
                m_stopButtonPressedPub.publish(std_msgs::Empty());
                break;
            default:
                ROS_ERROR_STREAM("Not handled buttons (" << static_cast<int>(payload.button) << ")");
                break;
        }
    }

    void handleImuData(Device source, const ImuDataPayload& payload)
    {
        sensor_msgs::Imu msg;
        msg.header.stamp = ros::Time::now();
        msg.orientation.x = 0;
        msg.orientation.y = 0;
        msg.orientation.z = 0;
        msg.orientation.w = 0;
        msg.orientation_covariance.assign(-1.0);
        msg.angular_velocity.x = payload.angularRateX;
        msg.angular_velocity.y = payload.angularRateY;
        msg.angular_velocity.z = payload.angularRateZ;
        msg.angular_velocity_covariance.assign(0.0);
        msg.linear_acceleration.x = payload.accelerationX;
        msg.linear_acceleration.y = payload.accelerationY;
        msg.linear_acceleration.z = payload.accelerationZ;
        msg.linear_acceleration_covariance.assign(0.0);

        m_imuPub.publish(msg);
    }

    void handleMotorStatus(Device source, const MotorStatusPayload& payload)
    {
        daemon_ros_client::MotorStatus msg;
        msg.header.stamp = ros::Time::now();

        msg.torso_orientation = payload.torsoOrientation;
        msg.torso_servo_speed = payload.torsoServoSpeed;

        msg.head_servo_angles[0] = payload.headServoAngle1;
        msg.head_servo_angles[1] = payload.headServoAngle2;
        msg.head_servo_angles[2] = payload.headServoAngle3;
        msg.head_servo_angles[3] = payload.headServoAngle4;
        msg.head_servo_angles[4] = payload.headServoAngle5;
        msg.head_servo_angles[5] = payload.headServoAngle6;
        msg.head_servo_speeds[0] = payload.headServoSpeed1;
        msg.head_servo_speeds[1] = payload.headServoSpeed2;
        msg.head_servo_speeds[2] = payload.headServoSpeed3;
        msg.head_servo_speeds[3] = payload.headServoSpeed4;
        msg.head_servo_speeds[4] = payload.headServoSpeed5;
        msg.head_servo_speeds[5] = payload.headServoSpeed6;
        msg.head_pose_frame_id = HEAD_POSE_FRAME_ID;
        msg.head_pose.position.x = payload.headPosePositionX;
        msg.head_pose.position.y = payload.headPosePositionY;
        msg.head_pose.position.z = payload.headPosePositionZ;
        msg.head_pose.orientation.x = payload.headPoseOrientationX;
        msg.head_pose.orientation.y = payload.headPoseOrientationY;
        msg.head_pose.orientation.z = payload.headPoseOrientationZ;
        msg.head_pose.orientation.w = payload.headPoseOrientationW;
        msg.is_head_pose_reachable = payload.isHeadPoseReachable;

        sendTorsoTf(msg.header.stamp, msg.torso_orientation);
        sendHeadTf(msg.header.stamp, msg.head_pose);
    }

    void sendTorsoTf(const ros::Time& stamp, float torsoOrientation)
    {
        geometry_msgs::TransformStamped transformStamped;

        transformStamped.header.stamp = stamp;
        transformStamped.header.frame_id = "base_link";
        transformStamped.child_frame_id = "torso_base";
        transformStamped.transform.translation.x = 0;
        transformStamped.transform.translation.y = 0;
        transformStamped.transform.translation.z = m_configuration.baseLinkTorsoBaseDeltaZ;

        tf2::Quaternion q;
        q.setRPY(0, 0, torsoOrientation);
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        m_tfBroadcaster.sendTransform(transformStamped);
    }

    void sendHeadTf(const ros::Time& stamp, const geometry_msgs::Pose& pose)
    {
        geometry_msgs::TransformStamped transformStamped;

        transformStamped.header.stamp = stamp;
        transformStamped.header.frame_id = HEAD_POSE_FRAME_ID;
        transformStamped.child_frame_id = "head";
        transformStamped.transform.translation.x = pose.position.x;
        transformStamped.transform.translation.y = pose.position.y;
        transformStamped.transform.translation.z = pose.position.z;

        transformStamped.transform.rotation.x = pose.orientation.x;
        transformStamped.transform.rotation.y = pose.orientation.y;
        transformStamped.transform.rotation.z = pose.orientation.z;
        transformStamped.transform.rotation.w = pose.orientation.w;

        m_tfBroadcaster.sendTransform(transformStamped);
    }





};

#endif