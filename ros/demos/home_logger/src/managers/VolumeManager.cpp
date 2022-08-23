#include "VolumeManager.h"

#include <cmath>

using namespace std;

VolumeManager::VolumeManager(ros::NodeHandle& nodeHandle) : m_currentVolumePercent(38)
{
    m_volumePublisher = nodeHandle.advertise<std_msgs::Int8>("opencr/audio_power_amplifier_volume", 1);
}

VolumeManager::~VolumeManager() {}

void VolumeManager::setVolume(float volume)
{
    volume = max(0.f, min(volume, 100.f));
    if (volume == 0)
    {
        mute();
        return;
    }
    m_currentVolumePercent = volume;
    m_isMuted = false;

    m_volumePublisher.publish(volumeToMsg(m_currentVolumePercent));
}

float VolumeManager::getVolume() const
{
    if (m_isMuted)
    {
        return 0.f;
    }
    else
    {
        return m_currentVolumePercent;
    }
}

bool VolumeManager::isMuted() const
{
    return m_isMuted;
}

void VolumeManager::mute()
{
    m_isMuted = true;
    m_volumePublisher.publish(volumeToMsg(0));
}

void VolumeManager::unmute()
{
    m_isMuted = true;
    m_volumePublisher.publish(volumeToMsg(m_currentVolumePercent));
}

std_msgs::Int8 VolumeManager::volumeToMsg(float volumePercent)
{
    std_msgs::Int8 msg;
    msg.data = static_cast<int8_t>(volumePercent / 100.f * 63);
    return msg;
}
