#ifndef MAIN_COMMON_H
#define MAIN_COMMON_H

#include "config.h"

#include "actuators/AudioPowerAmplifier.h"
#include "actuators/Buzzer.h"
#include "actuators/Fan.h"
#include "actuators/LedStrip.h"

#include "battery/Rrc20542Battery.h"
#include "battery/RrcPmm240Charger.h"

#include "sensors/AlsPt19LightSensor.h"
#include "sensors/CurrentVoltageSensor.h"
#include "sensors/PushButton.h"
#include "sensors/Thermistor.h"

#include <Arduino.h>
#include <Wire.h>

#define CRITICAL_ERROR(message)                                                                                        \
    while (true)                                                                                                       \
    {                                                                                                                  \
        DEBUG_SERIAL.println((message));                                                                               \
        delay(ERROR_DELAY_MS);                                                                                         \
    }

extern AudioPowerAmplifier audioPowerAmplifier;
extern Buzzer buzzer;
extern Fan fan;
extern LedStrip ledStrip;

extern Rrc20542Battery battery;
extern RrcPmm240Charger charger;

extern AlsPt19LightSensor frontLightSensor;
extern AlsPt19LightSensor backLightSensor;
extern AlsPt19LightSensor leftLightSensor;
extern AlsPt19LightSensor rightLightSensor;
extern CurrentVoltageSensor currentVoltageSensor;
extern PushButton startButton;
extern PushButton stopButton;
extern PushButton volumeUpButton;
extern PushButton volumeDownButton;
extern Thermistor onboardThermistor;
extern Thermistor externalThermistor;

void setupDebugSerial();
void setupWire();
void setupPwm();
void setupAdc();

void setupAudioPowerAmplifier();
void setupBuzzer();
void setupFan();
void setupLedStrip();

void setupCharger();

void setupLightSensors();
void setupCurrentVoltageSensor();
void setupPushButtons();
void setupThermistors();

#endif
