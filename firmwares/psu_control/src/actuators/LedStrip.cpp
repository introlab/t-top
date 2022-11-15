#include "LedStrip.h"

constexpr uint32_t RED_COLOR = 0xFF0000;
constexpr uint32_t ORANGE_COLOR = 0xE05800;
constexpr uint32_t YELLOW_COLOR = 0xFFFF00;
constexpr uint32_t GREEN_COLOR = 0x00FF00;

LedStrip::LedStrip(uint8_t pin) : m_leds(LED_COUNT, m_displayMemory, m_drawingMemory, pin, WS2812_GRB) {}

bool LedStrip::begin()
{
    memset(m_drawingMemory, 0, sizeof(m_drawingMemory));
    memset(m_displayMemory, 0, sizeof(m_displayMemory));
    return m_leds.begin();
}

void LedStrip::setStateOfCharge(float stateOfCharge)
{
    setLevel(0, stateOfCharge, 100.f, STATE_OF_CHARGE_LED_COUNT);
}

void LedStrip::setVolume(uint8_t volume)
{
    setLevel(
        STATE_OF_CHARGE_LED_COUNT,
        static_cast<float>(volume),
        static_cast<float>(AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME),
        VOLUME_LED_COUNT);
}

bool LedStrip::setBaseLedColors(const Color* colors, size_t size)
{
    if (size > BASE_LED_COUNT)
    {
        return false;
    }

    for (size_t i = 0; i < size; i++)
    {
        m_leds
            .setPixel(STATE_OF_CHARGE_LED_COUNT + VOLUME_LED_COUNT + i, colors[i].red, colors[i].green, colors[i].blue);
    }
    m_leds.show();

    return true;
}

void LedStrip::setLevel(size_t offset, float value, float maximumValue, size_t ledCount)
{
    const float step = maximumValue / ledCount;
    const float redThreshold = step / 4 - step / 8;
    const float orangeThreshold = redThreshold + step / 4;
    const float yellowThreshold = orangeThreshold + step / 4;
    const float greenThreshold = yellowThreshold + step / 4;

    for (size_t i = 0; i < STATE_OF_CHARGE_LED_COUNT; i++)
    {
        if (value > greenThreshold)
        {
            m_leds.setPixel(offset + i, GREEN_COLOR);
        }
        else if (value > yellowThreshold)
        {
            m_leds.setPixel(offset + i, YELLOW_COLOR);
        }
        else if (value > orangeThreshold)
        {
            m_leds.setPixel(offset + i, ORANGE_COLOR);
        }
        else if (value > redThreshold)
        {
            m_leds.setPixel(offset + i, RED_COLOR);
        }

        value -= step;
    }

    m_leds.show();
}
