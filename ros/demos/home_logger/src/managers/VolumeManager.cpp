#include "VolumeManager.h"

#include <cmath>

using namespace std;

constexpr float MAXIMUM_INT_VOLUME_VALUE = 63.f;

VolumeManager::VolumeManager(rclcpp::Node::SharedPtr node)
    : m_node(move(node)),
      m_currentVolumePercent(35),
      m_maximumVolumePercent(100.f)
{
    m_volumePublisher = node->create_publisher<std_msgs::msg::UInt8>("daemon/set_volume", 1);
    m_baseStatusSubscriber = node->create_subscription<daemon_ros_client::msg::BaseStatus>(
        "daemon/base_status",
        1,
        [this](const daemon_ros_client::msg::BaseStatus::SharedPtr msg) { baseStatusSubscriberCallback(msg); });

    setVolume(m_currentVolumePercent);
}

VolumeManager::~VolumeManager() {}

void VolumeManager::setVolume(float volume)
{
    volume = max(10.f, min(volume, m_maximumVolumePercent));
    m_currentVolumePercent = volume;

    m_volumePublisher->publish(volumeToMsg(m_currentVolumePercent));
}

float VolumeManager::getVolume() const
{
    return m_currentVolumePercent;
}

void VolumeManager::baseStatusSubscriberCallback(const daemon_ros_client::msg::BaseStatus::SharedPtr msg)
{
    m_currentVolumePercent = volumeIntToPercent(msg->volume);
    m_maximumVolumePercent = volumeIntToPercent(msg->maximum_volume);
}

std_msgs::msg::UInt8 VolumeManager::volumeToMsg(float volumePercent)
{
    std_msgs::msg::UInt8 msg;
    msg.data = volumePercentToInt(volumePercent);
    return msg;
}

float VolumeManager::volumeIntToPercent(uint8_t v)
{
    return static_cast<float>(v) / MAXIMUM_INT_VOLUME_VALUE * 100.f;
}

uint8_t VolumeManager::volumePercentToInt(float v)
{
    return static_cast<int8_t>(v / 100.f * MAXIMUM_INT_VOLUME_VALUE);
}
