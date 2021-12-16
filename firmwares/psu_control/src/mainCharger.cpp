#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_SETUP_BATTERY_CHARGER

#include "mainCommon.h"

constexpr uint32_t STARTUP_DELAY = 5000;
constexpr uint32_t STATUS_DELAY = 10000;

void setChargerCurrentLimit();

void printBatteryStatus();
void printChargerStatus();

void setup() {
  DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);

  delay(STARTUP_DELAY);
  initRandom();
  setupWire();
  setupCharger();

  setChargerCurrentLimit();
}

void loop() {
  printBatteryStatus();
  DEBUG_SERIAL.println();
  printChargerStatus();

  delay(STATUS_DELAY);
}


void setChargerCurrentLimit() {
  DEBUG_SERIAL.println("Set charger current limit - start");

  if (!charger.writeChargeCurrentLimitToEeprom(BATTERY_CHARGER_CHARGE_CURRENT_LIMIT)) {
    DEBUG_SERIAL.println("writeChargeCurrentLimitToEeprom failed");
  }
  if (!charger.writeChargeCurrentLimitToRam(BATTERY_CHARGER_CHARGE_CURRENT_LIMIT)) {
    DEBUG_SERIAL.println("writeChargeCurrentLimitToRam failed");
  }

  if (!charger.writeInputCurrentLimitToEeprom(BATTERY_CHARGER_INPUT_CURRENT_LIMIT)) {
    DEBUG_SERIAL.println("writeInputCurrentLimitToEeprom failed");
  }
  if (!charger.writeInputCurrentLimitToRam(BATTERY_CHARGER_INPUT_CURRENT_LIMIT)) {
    DEBUG_SERIAL.println("writeInputCurrentLimitToRam failed");
  }

  DEBUG_SERIAL.println("Set charger current limit - end");
}

void printBatteryStatus() {
  constexpr uint8_t MAX_NAME_SIZE = 128;
  char name[MAX_NAME_SIZE];
  int day, month, year;
  uint16_t serialNumber;
  uint16_t cycleCount;

  float temperature;
  float voltage;
  float current;
  float capacity;
  float stateOfCharge;
  float time;
  bool isFullyDischarged, isFullyCharged;
  RrcBatteryErrorCode error;

  DEBUG_SERIAL.println("**********************Battery Status**********************");
  if (battery.readManufacturerName(name, MAX_NAME_SIZE)) {
    DEBUG_SERIAL.print("Manufacturer Name: ");
    DEBUG_SERIAL.println(name);
  }
  if (battery.readDeviceName(name, MAX_NAME_SIZE)) {
    DEBUG_SERIAL.print("Device Name: ");
    DEBUG_SERIAL.println(name);
  }
  if (battery.readDeviceChemistry(name, MAX_NAME_SIZE)) {
    DEBUG_SERIAL.print("Device Chemistry: ");
    DEBUG_SERIAL.println(name);
  }
  if (battery.readManufactureDate(day, month, year)) {
    DEBUG_SERIAL.print("Manufacture Date (YYYY/MM/DD): ");
    DEBUG_SERIAL.print(year);
    DEBUG_SERIAL.print("/");
    DEBUG_SERIAL.print(month);
    DEBUG_SERIAL.print("/");
    DEBUG_SERIAL.println(day);
  }
  if (battery.readSerialNumber(serialNumber)) {
    DEBUG_SERIAL.print("Serial Number: ");
    DEBUG_SERIAL.println(serialNumber);
  }

  Serial.println();
  if (battery.readDesignVoltage(voltage)) {
    DEBUG_SERIAL.print("Design Voltage: ");
    DEBUG_SERIAL.print(voltage);
    DEBUG_SERIAL.println(" V");
  }
  if (battery.readDesignCapacity(capacity)) {
    DEBUG_SERIAL.print("Design Capacity: ");
    DEBUG_SERIAL.print(capacity);
    DEBUG_SERIAL.println(" Ah");
  }

  DEBUG_SERIAL.println();
  if (battery.readBatteryStatus(isFullyDischarged, isFullyCharged, error)) {
    DEBUG_SERIAL.print("Battey Status: isFullyDischarged=");
    DEBUG_SERIAL.print(isFullyDischarged);
    DEBUG_SERIAL.print(", isFullyCharged=");
    DEBUG_SERIAL.print(isFullyCharged);
    DEBUG_SERIAL.print(", error=");
    DEBUG_SERIAL.println(static_cast<int>(error));
  }
  if (battery.readCycleCount(cycleCount)) {
    DEBUG_SERIAL.print("Cycle Count: ");
    DEBUG_SERIAL.println(cycleCount);
  }
  if (battery.readTemperature(temperature)) {
    DEBUG_SERIAL.print("Temperature: ");
    DEBUG_SERIAL.print(temperature);
    DEBUG_SERIAL.println(" C");
  }
  if (battery.readVoltage(voltage)) {
    DEBUG_SERIAL.print("Voltage: ");
    DEBUG_SERIAL.print(voltage);
    DEBUG_SERIAL.println(" V");
  }
  if (battery.readCurrent(current)) {
    DEBUG_SERIAL.print("Current: ");
    DEBUG_SERIAL.print(current);
    DEBUG_SERIAL.println(" A");
  }
  if (battery.readAverageCurrent(current)) {
    DEBUG_SERIAL.print("Average Current: ");
    DEBUG_SERIAL.print(current);
    DEBUG_SERIAL.println(" A");
  }
  if (battery.readRelativeStateOfCharge(stateOfCharge)) {
    DEBUG_SERIAL.print("Relative State of Charge: ");
    DEBUG_SERIAL.print(stateOfCharge);
    DEBUG_SERIAL.println("%");
  }
  if (battery.readAbsoluteStateOfCharge(stateOfCharge)) {
    DEBUG_SERIAL.print("Absolute State of Charge: ");
    DEBUG_SERIAL.print(stateOfCharge);
    DEBUG_SERIAL.println("%");
  }
  if (battery.readRemainingCapacity(capacity)) {
    DEBUG_SERIAL.print("Remaining Capacity: ");
    DEBUG_SERIAL.print(capacity);
    DEBUG_SERIAL.println(" Ah");
  }
  if (battery.readFullChargeCapacity(capacity)) {
    DEBUG_SERIAL.print("Full Charge Capacity: ");
    DEBUG_SERIAL.print(capacity);
    DEBUG_SERIAL.println(" Ah");
  }
  if (battery.readFullChargeCapacity(capacity)) {
    DEBUG_SERIAL.print("Full Charge Capacity: ");
    DEBUG_SERIAL.print(capacity);
    DEBUG_SERIAL.println(" Ah");
  }
  if (battery.readRunTimeToEmpty(time)) {
    DEBUG_SERIAL.print("Runtime To Empty: ");
    DEBUG_SERIAL.print(time);
    DEBUG_SERIAL.println(" min");
  }
  if (battery.readAverageTimeToEmpty(time)) {
    DEBUG_SERIAL.print("Average Runtime To Empty: ");
    DEBUG_SERIAL.print(time);
    DEBUG_SERIAL.println(" min");
  }
  if (battery.readAverageTimeToFull(time)) {
    DEBUG_SERIAL.print("Average Runtime To Full: ");
    DEBUG_SERIAL.print(time);
    DEBUG_SERIAL.println(" min");
  }
}

void printChargerStatus() {
  DEBUG_SERIAL.println("**********************Charger Status**********************");

  DEBUG_SERIAL.print("isBatteryCharged=");
  DEBUG_SERIAL.println(charger.isBatteryCharged());

  DEBUG_SERIAL.print("isBatteryCharging=");
  DEBUG_SERIAL.println(charger.isBatteryCharging());

  DEBUG_SERIAL.print("hasBatteryError=");
  DEBUG_SERIAL.println(charger.hasBatteryError());

  DEBUG_SERIAL.print("isPsuConnected=");
  DEBUG_SERIAL.println(charger.isPsuConnected());
  DEBUG_SERIAL.print("hasChargerError=");
  DEBUG_SERIAL.println(charger.hasChargerError());
}

#endif
