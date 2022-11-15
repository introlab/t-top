#include "actuators/AudioPowerAmplifier.h"
#include "config.h"

AudioPowerAmplifier::AudioPowerAmplifier(TwoWire& wire, const uint8_t* addresses, const size_t size)
    : m_wire(wire),
      m_addresses(addresses),
      m_size(size),
      m_maximumVolume(AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME),
      m_volume(AUDIO_POWER_AMPLIFIER_DEFAULT_VOLUME)
{
}

void AudioPowerAmplifier::begin()
{
    setVolume(m_volume);
}

void AudioPowerAmplifier::setMaximumVolume(uint8_t maximumVolume)
{
    m_maximumVolume = maximumVolume;
    if (m_maximumVolume > AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME)
    {
        m_maximumVolume = AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME;
    }
    if (m_volume > m_maximumVolume)
    {
        setVolume(m_maximumVolume);
    }
}

void AudioPowerAmplifier::setVolume(uint8_t volume)
{
    m_volume = volume;
    if (m_volume > m_maximumVolume)
    {
        m_volume = m_maximumVolume;
    }

    for (size_t i = 0; i < m_size; i++)
    {
        writeVolume(m_addresses[i], m_volume);
    }
}

void AudioPowerAmplifier::writeVolume(int address, uint8_t volume)
{
    m_wire.beginTransmission(address);
    m_wire.write(volume);
    m_wire.endTransmission();
}
