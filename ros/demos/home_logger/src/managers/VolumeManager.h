#ifndef HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H
#define HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H

#include <ros/ros.h>
#include <std_msgs/UInt8.h>
#include <daemon_ros_client/BaseStatus.h>

#include <hbba_lite/utils/ClassMacros.h>

class VolumeManager
{
    float m_currentVolumePercent;
    float m_maximumVolumePercent;

    ros::Publisher m_volumePublisher;
    ros::Subscriber m_baseStatusSubscriber;

public:
    VolumeManager(ros::NodeHandle& nodeHandle);
    virtual ~VolumeManager();

    DECLARE_NOT_COPYABLE(VolumeManager);
    DECLARE_NOT_MOVABLE(VolumeManager);

    void setVolume(float volumePercent);
    float getVolume() const;

private:
    void baseStatusSubscriberCallback(const daemon_ros_client::BaseStatus::ConstPtr& msg);

    std_msgs::UInt8 volumeToMsg(float volumePercent);
    float volumeIntToPercent(uint8_t v);
    uint8_t volumePercentToInt(float v);
};

#endif
