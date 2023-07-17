#include "Thermistor.h"
#include "../config.h"

#include <Arduino.h>
#include <cmath>

using namespace std;

Thermistor::Thermistor(uint8_t pin, float ntcR, float ntcBeta, float r)
    : m_pin(pin),
      m_ntcR(ntcR),
      m_ntcBeta(ntcBeta),
      m_r(r)
{
}

void Thermistor::begin()
{
    pinMode(m_pin, INPUT);
}

float Thermistor::readCelsius()
{
    // Inspired by https://www.allaboutcircuits.com/projects/measuring-temperature-with-an-ntc-thermistor/
    float adcValue = analogRead(m_pin);
    float ntcR = m_r * ((ADC_MAX_VALUE / adcValue) - 1);

    return (m_ntcBeta * 298.15f) / (m_ntcBeta + (298.15f * log(ntcR / m_ntcR))) - 273.15f;
}
