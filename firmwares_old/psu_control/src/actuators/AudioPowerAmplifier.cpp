#include "actuators/AudioPowerAmplifier.h"
#include "config.h"

constexpr int AUDIO_POWER_AMPLIFIER_MAX_VOLUME = 63;

AudioPowerAmplifier::AudioPowerAmplifier(TwoWire& wire) : m_wire(wire) {}

AudioPowerAmplifier::~AudioPowerAmplifier() {}

void AudioPowerAmplifier::begin()
{
    setVolume(24);
}

void AudioPowerAmplifier::setVolume(uint8_t volume)
{
    if (volume > AUDIO_POWER_AMPLIFIER_MAX_VOLUME)
    {
        volume = AUDIO_POWER_AMPLIFIER_MAX_VOLUME;
    }

    for (int i = 0; i < AUDIO_POWER_AMPLIFIER_COUNT; i++)
    {
        writeVolume(AUDIO_POWER_AMPLIFIER_I2C_ADDRESSES[i], volume);
    }
}

void AudioPowerAmplifier::writeVolume(int address, uint8_t volume)
{
    m_wire.beginTransmission(address);
    m_wire.write(volume);
    m_wire.endTransmission();
}
