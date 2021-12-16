#include "battery/Rrc20542Battery.h"

Rrc20542Battery::Rrc20542Battery(TwoWire& wire) : m_wire(wire) {}

bool Rrc20542Battery::readTemperature(float& temperature, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x08, value, trialCount)) { return false; }

  temperature = value;
  temperature /= 10;
  temperature -= 273.15;
  return true;
}

bool Rrc20542Battery::readVoltage(float& voltage, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x09, value, trialCount)) { return false; }

  voltage = value;
  voltage /= 1000;
  return true;
}

bool Rrc20542Battery::readCurrent(float& current, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x0a, value, trialCount)) { return false; }

  current = value;
  current /= 1000;
  return true;
}

bool Rrc20542Battery::readAverageCurrent(float& current, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x0b, value, trialCount)) { return false; }

  current = value;
  current /= 1000;
  return true;
}

bool Rrc20542Battery::readRelativeStateOfCharge(float& stateOfCharge, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x0d, value, trialCount)) { return false; }

  stateOfCharge = value;
  return true;
}

bool Rrc20542Battery::readAbsoluteStateOfCharge(float& stateOfCharge, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x0e, value, trialCount)) { return false; }

  stateOfCharge = value;
  return true;
}

bool Rrc20542Battery::readRemainingCapacity(float& capacity, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x0f, value, trialCount)) { return false; }

  capacity = value;
  capacity /= 1000;
  return true;
}

bool Rrc20542Battery::readFullChargeCapacity(float& capacity, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x10, value, trialCount)) { return false; }

  capacity = value;
  capacity /= 1000;
  return true;
}

bool Rrc20542Battery::readRunTimeToEmpty(float& time, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x11, value, trialCount)) { return false; }

  time = value;
  return true;
}

bool Rrc20542Battery::readAverageTimeToEmpty(float& time, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x12, value, trialCount)) { return false; }

  time = value;
  return true;
}

bool Rrc20542Battery::readAverageTimeToFull(float& time, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x13, value, trialCount)) { return false; }

  time = value;
  return true;
}

bool Rrc20542Battery::readBatteryStatus(bool& isFullyDischarged, bool& isFullyCharged, RrcBatteryErrorCode& error, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x16, value, trialCount)) { return false; }

  isFullyDischarged = (value & 0b0000'0000'0001'0000) != 0;
  isFullyCharged = (value & 0b0000'0000'0010'0000) != 0;
  error = static_cast<RrcBatteryErrorCode>(value & 0b0000'0000'0000'1111);
  return true;
}

bool Rrc20542Battery::readCycleCount(uint16_t& cycleCount, size_t trialCount) {
  return readWordTrials(0x17, cycleCount, trialCount);
}

bool Rrc20542Battery::readDesignCapacity(float& capacity, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x18, value, trialCount)) { return false; }

  capacity = value;
  capacity /= 1000;
  return true;
}

bool Rrc20542Battery::readDesignVoltage(float& voltage, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x19, value, trialCount)) { return false; }

  voltage = value;
  voltage /= 1000;
  return true;
}

bool Rrc20542Battery::readManufacturerName(char* name, uint8_t maxNameSize, size_t trialCount) {
  if (maxNameSize == 0) { return false; }

  uint8_t size = 0;
  if (!readBlockTrials(0x20, maxNameSize - 1, name, size, trialCount)) { return false; }

  name[size] = '\0';
  return true;
}

bool Rrc20542Battery::readDeviceName(char* name, uint8_t maxNameSize, size_t trialCount) {
  if (maxNameSize == 0) { return false; }

  uint8_t size = 0;
  if (!readBlockTrials(0x21, maxNameSize - 1, name, size, trialCount)) { return false; }

  name[size] = '\0';
  return true;
}

bool Rrc20542Battery::readDeviceChemistry(char* name, uint8_t maxNameSize, size_t trialCount) {
  if (maxNameSize == 0) { return false; }

  uint8_t size = 0;
  if (!readBlockTrials(0x22, maxNameSize - 1, name, size, trialCount)) { return false; }

  name[size] = '\0';
  return true;
}

bool Rrc20542Battery::readManufactureDate(int& day, int& month, int& year, size_t trialCount) {
  uint16_t value;
  if (!readWordTrials(0x1b, value, trialCount)) { return false; }

  day = value & 0b0000'0000'0001'1111;
  month = (value & 0b0000'0001'1110'0000) >> 5;
  year = ((value & 0b1111'1110'0000'0000) >> 9) + 1980;
  return true;
}

bool Rrc20542Battery::readSerialNumber(uint16_t& serialNumber, size_t trialCount) {
  return readWordTrials(0x1c, serialNumber, trialCount);
}

bool Rrc20542Battery::readWord(uint8_t command, uint16_t& value) {
  m_wire.beginTransmission(RRC_20542_BATTERY_ADDRESS);
  m_wire.write(command);
  if (m_wire.endTransmission(false) != 0) { return false; }

  if (m_wire.requestFrom(RRC_20542_BATTERY_ADDRESS, sizeof(RrcWordUnion)) != sizeof(RrcWordUnion)) {
    return false;
  }

  RrcWordUnion word;
  word.bytes[0] = m_wire.read();
  word.bytes[1] = m_wire.read();
  value = word.word;

  return true;
}

bool Rrc20542Battery::readBlock(uint8_t command, uint8_t maxSize, char* data, uint8_t& size) {
  constexpr size_t MAX_BLOCK_SIZE = 33;

  m_wire.beginTransmission(RRC_20542_BATTERY_ADDRESS);
  m_wire.write(command);
  if (m_wire.endTransmission(false) != 0) { return false; }

  size_t receivedByteCount = m_wire.requestFrom(RRC_20542_BATTERY_ADDRESS, MAX_BLOCK_SIZE);
  if (receivedByteCount < 1) { return false; }

  size = m_wire.read();
  if (size > maxSize) { return false; }

  for (size_t i = 0; i < receivedByteCount - 1; i++) {
    char value = m_wire.read();
    if (i < size) {
      data[i] = value;
    }
  }

  return true;
}

bool Rrc20542Battery::readWordTrials(uint8_t command, uint16_t& value, size_t trialCount)
{
  for (size_t i = 0; i < trialCount; i++) {
    if (readWord(command, value)) {
      return true;
    }
    smBusRandomDelay();
  }
  return false;
}

bool Rrc20542Battery::readBlockTrials(uint8_t command, uint8_t maxSize, char* data, uint8_t& size, size_t trialCount)
{
  for (size_t i = 0; i < trialCount; i++) {
    if (readBlock(command, maxSize, data, size)) {
      return true;
    }
    smBusRandomDelay();
  }
  return false;
}
