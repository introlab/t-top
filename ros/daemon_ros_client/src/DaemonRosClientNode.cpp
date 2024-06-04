#include "DaemonRosClientNode.h"
#include "QtUtils.h"


DaemonRosClientNode::DaemonRosClientNode() : rclcpp::Node("daemon_ros_client_node"), m_tfBroadcaster(*this)
{
    m_baseLinkTorsoBaseDeltaZ = declare_parameter("base_link_torso_base_delta_z", 0.0);

    initROS();
    initWebSocketProtocolWrapper();
}

void DaemonRosClientNode::initROS()
{
    m_baseStatusPub = create_publisher<daemon_ros_client::msg::BaseStatus>("daemon/base_status", PubQueueSize);
    m_startButtonPressedPub = create_publisher<std_msgs::msg::Empty>("daemon/start_button_pressed", PubQueueSize);
    m_stopButtonPressedPub = create_publisher<std_msgs::msg::Empty>("daemon/stop_button_pressed", PubQueueSize);
    m_imuPub = create_publisher<sensor_msgs::msg::Imu>("daemon/imu/data_raw", PubQueueSize);
    m_motorStatusPub = create_publisher<daemon_ros_client::msg::MotorStatus>("daemon/motor_status", PubQueueSize);

    m_setVolumeSub = create_subscription<std_msgs::msg::UInt8>(
        "daemon/set_volume",
        SubQueueSize,
        [this](const std_msgs::msg::UInt8::SharedPtr msg) { setVolumeCallback(msg); });
    m_setLedColorsSub = create_subscription<daemon_ros_client::msg::LedColors>(
        "daemon/set_led_colors",
        SubQueueSize,
        [this](const daemon_ros_client::msg::LedColors::SharedPtr msg) { setLedColorsCallback(msg); });
    m_setTorsoOrientationSub = create_subscription<std_msgs::msg::Float32>(
        "daemon/set_torso_orientation",
        SubQueueSize,
        [this](const std_msgs::msg::Float32::SharedPtr msg) { setTorsoOrientationCallback(msg); });
    m_setHeadPoseSub = create_subscription<geometry_msgs::msg::PoseStamped>(
        "daemon/set_head_pose",
        SubQueueSize,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { setHeadPoseCallback(msg); });
}

void DaemonRosClientNode::initWebSocketProtocolWrapper()
{
    QUrl url(WebSocketProtocolWrapper::ROS_DEFAULT_CLIENT_URL);
    m_websocketProtocolWrapper = new WebSocketProtocolWrapper(url, this);

    // Connect only useful signals
    // WARNING Using direct connection, handleXXX functions must be thread safe)
    connect(
        m_websocketProtocolWrapper,
        &WebSocketProtocolWrapper::newBaseStatus,
        this,
        &DaemonRosClientNode::handleBaseStatus,
        Qt::DirectConnection);
    connect(
        m_websocketProtocolWrapper,
        &WebSocketProtocolWrapper::newButtonPressed,
        this,
        &DaemonRosClientNode::handleButtonPressed,
        Qt::DirectConnection);
    connect(
        m_websocketProtocolWrapper,
        &WebSocketProtocolWrapper::newImuData,
        this,
        &DaemonRosClientNode::handleImuData,
        Qt::DirectConnection);
    connect(
        m_websocketProtocolWrapper,
        &WebSocketProtocolWrapper::newMotorStatus,
        this,
        &DaemonRosClientNode::handleMotorStatus,
        Qt::DirectConnection);
}

void DaemonRosClientNode::setVolumeCallback(const std_msgs::msg::UInt8::SharedPtr& msg)
{
    SetVolumePayload payload;
    payload.volume = msg->data;

    invokeLater(
        [=]()
        {
            if (m_websocketProtocolWrapper)
            {
                m_websocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload);
            }
        });
}

void DaemonRosClientNode::setLedColorsCallback(const daemon_ros_client::msg::LedColors::SharedPtr& msg)
{
    SetLedColorsPayload payload;
    for (size_t i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        payload.colors[i].red = msg->colors[i].red;
        payload.colors[i].green = msg->colors[i].green;
        payload.colors[i].blue = msg->colors[i].blue;
    }

    invokeLater(
        [=]()
        {
            if (m_websocketProtocolWrapper)
            {
                m_websocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload);
            }
        });
}

void DaemonRosClientNode::setTorsoOrientationCallback(const std_msgs::msg::Float32::SharedPtr& msg)
{
    SetTorsoOrientationPayload payload;
    payload.torsoOrientation = msg->data;

    invokeLater(
        [=]()
        {
            if (m_websocketProtocolWrapper)
            {
                m_websocketProtocolWrapper->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload);
            }
        });
}

void DaemonRosClientNode::setHeadPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr& msg)
{
    if (msg->header.frame_id != HEAD_POSE_FRAME_ID)
    {
        RCLCPP_ERROR_STREAM(get_logger(), "Invalid head pose frame id (" << msg->header.frame_id << ")");
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

    invokeLater(
        [=]()
        {
            if (m_websocketProtocolWrapper)
            {
                m_websocketProtocolWrapper->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload);
            }
        });
}

void DaemonRosClientNode::handleBaseStatus(Device source, const BaseStatusPayload& payload)
{
    // WARNING must be tread safe, called from Qt Thread
    daemon_ros_client::msg::BaseStatus msg;
    msg.header.stamp = get_clock()->now();
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

    m_baseStatusPub->publish(msg);
}

void DaemonRosClientNode::handleButtonPressed(Device source, const ButtonPressedPayload& payload)
{
    // WARNING must be tread safe, called from Qt Thread
    switch (payload.button)
    {
        case Button::START:
            m_startButtonPressedPub->publish(std_msgs::msg::Empty());
            break;
        case Button::STOP:
            m_stopButtonPressedPub->publish(std_msgs::msg::Empty());
            break;
        default:
            RCLCPP_ERROR_STREAM(get_logger(), "Not handled buttons (" << static_cast<int>(payload.button) << ")");
            break;
    }
}

void DaemonRosClientNode::handleImuData(Device source, const ImuDataPayload& payload)
{
    // WARNING must be tread safe, called from Qt Thread
    sensor_msgs::msg::Imu msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = "dynamixel_control_imu";
    msg.orientation.x = 0;
    msg.orientation.y = 0;
    msg.orientation.z = 0;
    msg.orientation.w = 0;
    msg.orientation_covariance.fill(-1.0);
    msg.angular_velocity.x = payload.angularRateX;
    msg.angular_velocity.y = payload.angularRateY;
    msg.angular_velocity.z = payload.angularRateZ;
    msg.angular_velocity_covariance.fill(0.0);
    msg.linear_acceleration.x = payload.accelerationX;
    msg.linear_acceleration.y = payload.accelerationY;
    msg.linear_acceleration.z = payload.accelerationZ;
    msg.linear_acceleration_covariance.fill(0.0);

    m_imuPub->publish(msg);
}

void DaemonRosClientNode::handleMotorStatus(Device source, const MotorStatusPayload& payload)
{
    // WARNING must be tread safe, called from Qt Thread
    daemon_ros_client::msg::MotorStatus msg;
    msg.header.stamp = get_clock()->now();

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
    m_motorStatusPub->publish(msg);
}

void DaemonRosClientNode::sendTorsoTf(const rclcpp::Time& stamp, float torsoOrientation)
{
    geometry_msgs::msg::TransformStamped transformStamped;

    transformStamped.header.stamp = stamp;
    transformStamped.header.frame_id = "base_link";
    transformStamped.child_frame_id = "torso_base";
    transformStamped.transform.translation.x = 0;
    transformStamped.transform.translation.y = 0;
    transformStamped.transform.translation.z = m_baseLinkTorsoBaseDeltaZ;

    tf2::Quaternion q;
    q.setRPY(0, 0, torsoOrientation);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    m_tfBroadcaster.sendTransform(transformStamped);
}

void DaemonRosClientNode::sendHeadTf(const rclcpp::Time& stamp, const geometry_msgs::msg::Pose& pose)
{
    geometry_msgs::msg::TransformStamped transformStamped;

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

void DaemonRosClientNode::cleanup()
{
    SetLedColorsPayload payload;
    for (size_t i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        payload.colors[i].red = 0;
        payload.colors[i].green = 0;
        payload.colors[i].blue = 0;
    }

    if (m_websocketProtocolWrapper)
    {
        m_websocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload);
    }
}
