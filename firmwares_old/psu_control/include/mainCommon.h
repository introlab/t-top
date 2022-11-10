#ifndef MAIN_COMMON_H
#define MAIN_COMMON_H

#include "config.h"

#include "battery/Rrc20542Battery.h"
#include "battery/RrcPmm240Charger.h"
#include "battery/BatteryLed.h"
#include "battery/Buzzer.h"

#include "sensors/Temperature.h"
#include "actuators/Fan.h"

#include "actuators/AudioPowerAmplifier.h"
#include "sensors/CurrentVoltageSensor.h"

#include "OpencrCommandHandler.h"
#include "OpencrCommandSender.h"

#include <Arduino.h>
#include <Wire.h>

extern Rrc20542Battery battery;
extern RrcPmm240Charger charger;
extern BatteryLed batteryLed;
extern Buzzer buzzer;

extern Temperature onboardTemperature;
extern Temperature externalTemperature;
extern Fan fan;

extern CurrentVoltageSensor currentVoltageSensor;

void setupWire();
void setupPwm();
void setupAdc();
void setupCharger();
void setupBatteryLed();
void setupBuzzer();
void setupFan();
void setupCurrentVoltageSensor();

#endif
