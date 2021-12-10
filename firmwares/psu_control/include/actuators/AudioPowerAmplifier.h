#ifndef AUDIO_POWER_AMPLIFIER_H
#define AUDIO_POWER_AMPLIFIER_H

#include <Arduino.h>
#include <Wire.h>

class AudioPowerAmplifier {
  TwoWire& m_wire;

public:
  AudioPowerAmplifier(TwoWire& wire);
  ~AudioPowerAmplifier();

  void begin();
  void setVolume(uint8_t volume);

private:
  void writeVolume(int address, uint8_t volume);
};

#endif
