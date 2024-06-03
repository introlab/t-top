#ifndef HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H
#define HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <daemon_ros_client/msg/base_status.hpp>

#include <hbba_lite/utils/ClassMacros.h>

class VolumeManager
{
    rclcpp::Node::SharedPtr m_node;

    float m_currentVolumePercent;
    float m_maximumVolumePercent;

    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr m_volumePublisher;
    rclcpp::Subscription<daemon_ros_client::msg::BaseStatus>::SharedPtr m_baseStatusSubscriber;

public:
    VolumeManager(rclcpp::Node::SharedPtr node);
    virtual ~VolumeManager();

    DECLARE_NOT_COPYABLE(VolumeManager);
    DECLARE_NOT_MOVABLE(VolumeManager);

    void setVolume(float volumePercent);
    float getVolume() const;

private:
    void baseStatusSubscriberCallback(const daemon_ros_client::msg::BaseStatus::SharedPtr msg);

    std_msgs::msg::UInt8 volumeToMsg(float volumePercent);
    float volumeIntToPercent(uint8_t v);
    uint8_t volumePercentToInt(float v);
};

#endif
