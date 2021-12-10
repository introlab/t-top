#include "mainCommon.h"

Rrc20542Battery battery(BATTERY_WIRE);
RrcPmm240Charger charger(BATTERY_WIRE, BATTERY_STATUS_PIN, CHARGER_STATUS_PIN);
BatteryLed batteryLed;
Buzzer buzzer;

Temperature onboardTemperature(ONBOARD_TEMPERATURE_ADC_CHANNEL, ONBOARD_TEMPERATURE_NTC_R, ONBOARD_TEMPERATURE_NTC_BETA, ONBOARD_TEMPERATURE_R);
Temperature externalTemperature(EXTERNAL_TEMPERATURE_ADC_CHANNEL, EXTERNAL_TEMPERATURE_NTC_R, EXTERNAL_TEMPERATURE_NTC_BETA, EXTERNAL_TEMPERATURE_R);
Fan fan;

CurrentVoltageSensor currentVoltageSensor(CURRENT_VOLTAGE_SENSOR_WIRE);

void setupWire() {
  DEBUG_SERIAL.println("Setup Wire - start");
  Wire.setClock(WIRE_CLOCK);
  Wire1.setClock(WIRE_CLOCK);
  Wire.begin();
  Wire1.begin();
  DEBUG_SERIAL.println("Setup Wire - end");
}

void setupPwm() {
  DEBUG_SERIAL.println("Setup PWM - start");
  analogWriteResolution(PWM_RESOLUTION);
  DEBUG_SERIAL.println("Setup PWM - end");
}

void setupAdc() {
  DEBUG_SERIAL.println("Setup ADC - start");
  analogReadResolution(ADC_RESOLUTION);
  DEBUG_SERIAL.println("Setup ADC - end");
}

void setupCharger() {
  DEBUG_SERIAL.println("Setup Charger - start");
  charger.begin();
  DEBUG_SERIAL.println("Setup Charger - end");
}

void setupBatteryLed() {
  DEBUG_SERIAL.println("Setup Battery LED - start");
  batteryLed.begin();
  DEBUG_SERIAL.println("Setup Battery LED - end");
}

void setupBuzzer() {
  DEBUG_SERIAL.println("Setup Buzzer - start");
  buzzer.begin();
  DEBUG_SERIAL.println("Setup Buzzer - end");
}

void setupFan() {
  DEBUG_SERIAL.println("Setup FAN - start");
  fan.begin();
  DEBUG_SERIAL.println("Setup FAN - end");
}

void setupCurrentVoltageSensor() {
  DEBUG_SERIAL.println("Setup Current Voltage Sensor - start");
  if (!currentVoltageSensor.begin()) {
    DEBUG_SERIAL.println("begin() failed");
  }
  DEBUG_SERIAL.println("Setup Current Voltage Sensor - end");
}
