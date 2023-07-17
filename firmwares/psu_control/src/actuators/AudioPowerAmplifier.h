#ifndef PSU_CONTROL_ACTUATORS_AUDIO_POWER_AMPLIFIER_H
#define PSU_CONTROL_ACTUATORS_AUDIO_POWER_AMPLIFIER_H

#include <ClassMacro.h>

#include <Arduino.h>
#include <Wire.h>

class AudioPowerAmplifier
{
    TwoWire& m_wire;
    const uint8_t* m_addresses;
    const size_t m_size;

    uint8_t m_maximumVolume;
    uint8_t m_volume;

public:
    AudioPowerAmplifier(TwoWire& wire, const uint8_t* addresses, const size_t size);

    DECLARE_NOT_COPYABLE(AudioPowerAmplifier);
    DECLARE_NOT_MOVABLE(AudioPowerAmplifier);

    void begin();

    void setMaximumVolume(uint8_t maximumVolume);
    uint8_t maximumVolume() const;

    void setVolume(uint8_t volume);
    uint8_t volume() const;

private:
    void writeVolume(int address, uint8_t volume);
};

inline uint8_t AudioPowerAmplifier::maximumVolume() const
{
    return m_maximumVolume;
}

inline uint8_t AudioPowerAmplifier::volume() const
{
    return m_volume;
}

#endif
