#ifndef PSU_CONTROL_ACTUATORS_LED_STRIP_H
#define PSU_CONTROL_ACTUATORS_LED_STRIP_H

#include "../config.h"

#include <ClassMacro.h>
#include <SerialCommunication.h>

#include <WS2812Serial.h>

class LedStrip
{
    // State of charge, volume, base
    uint8_t m_drawingMemory[3 * LED_COUNT];
    uint8_t m_displayMemory[12 * LED_COUNT];

    WS2812Serial m_leds;

public:
    LedStrip(uint8_t pin);

    DECLARE_NOT_COPYABLE(LedStrip);
    DECLARE_NOT_MOVABLE(LedStrip);

    bool begin();

    void setBrightness(uint8_t brightness);

    void setStateOfCharge(float stateOfCharge);
    void setVolume(uint8_t volume);
    bool setBaseLedColors(const Color* colors, size_t size);

private:
    void setLevel(size_t offset, float value, float maximumValue, size_t ledCount);
};

inline void LedStrip::setBrightness(uint8_t brightness)
{
    m_leds.setBrightness(brightness);
    m_leds.show();
}


#endif
