#ifndef HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H
#define HOME_LOGGER_MANAGERS_VOLUME_MANAGER_H

#include <ros/ros.h>
#include <std_msgs/Int8.h>

class VolumeManager
{
    float m_currentVolumePercent;

    ros::Publisher m_volumePublisher;

public:
    VolumeManager(ros::NodeHandle& nodeHandle);
    virtual ~VolumeManager();

    void setVolume(float volumePercent);
    float getVolume() const;

private:
    std_msgs::Int8 volumeToMsg(float volumePercent);
};

#endif
