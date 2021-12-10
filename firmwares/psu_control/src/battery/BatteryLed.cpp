#include "battery/BatteryLed.h"
#include "config.h"

#include <Arduino.h>

BatteryLed::BatteryLed() {}

void BatteryLed::begin() {
  pinMode(BATTERY_25_LED_PIN, OUTPUT);
  pinMode(BATTERY_50_LED_PIN, OUTPUT);
  pinMode(BATTERY_75_LED_PIN, OUTPUT);
  pinMode(BATTERY_100_LED_PIN, OUTPUT);

  analogWriteFrequency(BATTERY_25_LED_PIN, BATTERY_LED_PWM_FREQUENCY);
  analogWriteFrequency(BATTERY_50_LED_PIN, BATTERY_LED_PWM_FREQUENCY);
  analogWriteFrequency(BATTERY_75_LED_PIN, BATTERY_LED_PWM_FREQUENCY);
  analogWriteFrequency(BATTERY_100_LED_PIN, BATTERY_LED_PWM_FREQUENCY);

  setStateOfCharge(0.0);
}

void BatteryLed::setStateOfCharge(float stateOfCharge) {
  if (stateOfCharge < 25.f) {
    analogWrite(BATTERY_25_LED_PIN, static_cast<int>(stateOfCharge / 25.f * PWM_MAX_VALUE));
    analogWrite(BATTERY_50_LED_PIN, 0);
    analogWrite(BATTERY_75_LED_PIN, 0);
    analogWrite(BATTERY_100_LED_PIN, 0);
  }
  else if (stateOfCharge < 50.f) {
    analogWrite(BATTERY_25_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_50_LED_PIN, static_cast<int>((stateOfCharge - 25.f) / 25.f * PWM_MAX_VALUE));
    analogWrite(BATTERY_75_LED_PIN, 0);
    analogWrite(BATTERY_100_LED_PIN, 0);
  }
  else if (stateOfCharge < 75.f) {
    analogWrite(BATTERY_25_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_50_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_75_LED_PIN, static_cast<int>((stateOfCharge - 50.f) / 25.f * PWM_MAX_VALUE));
    analogWrite(BATTERY_100_LED_PIN, 0);
  }
  else if (stateOfCharge < 100.f) {
    analogWrite(BATTERY_25_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_50_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_75_LED_PIN, PWM_MAX_VALUE);
    analogWrite(BATTERY_100_LED_PIN, static_cast<int>((stateOfCharge - 75.f) / 25.f * PWM_MAX_VALUE));
  }
}
