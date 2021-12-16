#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_TEST

#include "mainCommon.h"

constexpr uint32_t TEST_DELAY_MS = 5000;

void setup() {
  DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);
  OPENCR_SERIAL.begin(OPENCR_SERIAL_BAUD_RATE);

  initRandom();
  setupWire();
  setupPwm();
  setupAdc();

  setupCharger();
  setupBatteryLed();
  setupBuzzer();

  setupFan();

  setupCurrentVoltageSensor();
}

void testBatteryLed();
void testBuzzer();
void testTemperature();
void testFan();
void testCurrentVoltageSensor();

void loop() {
  testBatteryLed();
  testBuzzer();
  testTemperature();
  testFan();
  testCurrentVoltageSensor();

  delay(TEST_DELAY_MS);
}

void testBatteryLed() {
  DEBUG_SERIAL.println("---------------------Test Battery LEDs---------------------");

  for (float stateOfCharge = 0.f; stateOfCharge <= 100.f; stateOfCharge += 5) {
    DEBUG_SERIAL.print("State of Charge: ");
    DEBUG_SERIAL.println(stateOfCharge);
    batteryLed.setStateOfCharge(stateOfCharge);
    delay(1000);
  }

  delay(TEST_DELAY_MS);
}

void testBuzzer() {
  DEBUG_SERIAL.println("------------------------Test Buzzer------------------------");

  DEBUG_SERIAL.println("Enabling the buzzer");
  buzzer.enable();
  delay(5000);
  DEBUG_SERIAL.println("Disabling the buzzer");
  buzzer.disable();

  delay(TEST_DELAY_MS);
}

void testTemperature() {
  DEBUG_SERIAL.println("----------------------Test Temperature---------------------");
  DEBUG_SERIAL.print("Onboard Temperature: ");
  DEBUG_SERIAL.print(onboardTemperature.readCelsius());
  DEBUG_SERIAL.println(" C");

  DEBUG_SERIAL.print("External Temperature: ");
  DEBUG_SERIAL.print(externalTemperature.readCelsius());
  DEBUG_SERIAL.println(" C");
  delay(TEST_DELAY_MS);
}

void testFan() {
  DEBUG_SERIAL.println("--------------------------Test Fan-------------------------");

  DEBUG_SERIAL.println("Half Speed");
  fan.update(FAN_TEMPERATURE_STEP_1 + FAN_HYSTERESIS);
  delay(TEST_DELAY_MS);

  DEBUG_SERIAL.println("Full Speed");
  fan.update(FAN_TEMPERATURE_STEP_2 + FAN_HYSTERESIS);
  delay(TEST_DELAY_MS);

  DEBUG_SERIAL.println("Stop");
  fan.update(FAN_TEMPERATURE_STEP_1 - FAN_HYSTERESIS);
  delay(TEST_DELAY_MS);
}

void testCurrentVoltageSensor() {
  DEBUG_SERIAL.println("----------------Test Current/Voltage Sensor----------------");

  DEBUG_SERIAL.print("Current");
  DEBUG_SERIAL.print(currentVoltageSensor.readCurrent());
  DEBUG_SERIAL.println(" A");

  DEBUG_SERIAL.print("Voltage");
  DEBUG_SERIAL.print(currentVoltageSensor.readVoltage());
  DEBUG_SERIAL.println(" V");

  delay(TEST_DELAY_MS);
}


#endif
