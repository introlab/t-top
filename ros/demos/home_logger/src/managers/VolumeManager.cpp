#include "VolumeManager.h"

#include <cmath>

using namespace std;

VolumeManager::VolumeManager(ros::NodeHandle& nodeHandle) : m_currentVolumePercent(35)
{
    m_volumePublisher = nodeHandle.advertise<std_msgs::Int8>("opencr/audio_power_amplifier_volume", 1);
    setVolume(m_currentVolumePercent);
}

VolumeManager::~VolumeManager() {}

void VolumeManager::setVolume(float volume)
{
    volume = max(10.f, min(volume, 100.f));
    m_currentVolumePercent = volume;

    m_volumePublisher.publish(volumeToMsg(m_currentVolumePercent));
}

float VolumeManager::getVolume() const
{
    return m_currentVolumePercent;
}

std_msgs::Int8 VolumeManager::volumeToMsg(float volumePercent)
{
    std_msgs::Int8 msg;
    msg.data = static_cast<int8_t>(volumePercent / 100.f * 63);
    return msg;
}
