#ifndef DYNAMIXEL_CONTROL_MAIN_COMMON_H
#define DYNAMIXEL_CONTROL_MAIN_COMMON_H

#include "config.h"
#include "criticalError.h"

#include <Dynamixel2Arduino.h>

#include <Arduino.h>
#include <Wire.h>

extern Icm40627 imu;
extern Dynamixel2Arduino dynamixel;

void setupDebugSerial();
void setupWire();

void setupImu();
void setupImu(void (*dataReadyInterrupt)());

void setupDynamixel();

#endif
