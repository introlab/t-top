#include "sensors/Temperature.h"
#include "config.h"

#include <Arduino.h>
#include <cmath>

using namespace std;

Temperature::Temperature(uint8_t adcChannel, float ntcR, float ntcBeta, float r) :
  m_adcChannel(adcChannel),
  m_ntcR(ntcR),
  m_ntcBeta(ntcBeta),
  m_r(r) {}

float Temperature::readCelsius() {
  // Inspired by https://www.allaboutcircuits.com/projects/measuring-temperature-with-an-ntc-thermistor/
  float acdValue = analogRead(m_adcChannel);
  float ntcR = m_r * ((ADC_MAX_VALUE / acdValue) - 1);

  return (m_ntcBeta * 298.15) / (m_ntcBeta + (298.15 * log(ntcR / m_ntcR))) - 273.15;
}
