#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_NORMAL

#include "mainCommon.h"
#include "OpencrCommandHandler.h"
#include "OpencrCommandSender.h"

#include <Ticker.h>

AudioPowerAmplifier audioPowerAmplifier(AUDIO_POWER_AMPLIFIER_WIRE);

OpencrCommandHandler opencrCommandHandler;
OpencrCommandSender opencrCommandSender;

void setupAudioPowerAmplifier();
void setupOpencrCommandHandler();

void onVolumeCommand(uint8_t volume);
void onStatusTicker();

Ticker statusTicker(onStatusTicker, STATUS_TICKER_INTERVAL_MS, 0, MILLIS);

void setup()
{
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

    setupAudioPowerAmplifier();
    setupOpencrCommandHandler();

    onStatusTicker();
    statusTicker.start();
}

void loop()
{
    opencrCommandHandler.update();
    statusTicker.update();
}

void setupAudioPowerAmplifier()
{
    DEBUG_SERIAL.println("Setup Audio Power Amplifier - start");
    audioPowerAmplifier.begin();
    DEBUG_SERIAL.println("Setup Audio Power Amplifier - end");
}

void setupOpencrCommandHandler()
{
    DEBUG_SERIAL.println("Setup OpenCR Command Handler - start");
    opencrCommandHandler.setVolumeCommandHandler(&onVolumeCommand);
    DEBUG_SERIAL.println("Setup OpenCR Command Handler - end");
}

void onVolumeCommand(uint8_t volume)
{
    audioPowerAmplifier.setVolume(volume);
}

void onStatusTicker()
{
    float stateOfCharge = 0.f;
    if (!battery.readRelativeStateOfCharge(stateOfCharge))
    {
        DEBUG_SERIAL.println("Read relative state of charge failed");
        return;
    }
    batteryLed.setStateOfCharge(stateOfCharge);

    bool isBatteryCharging = charger.isBatteryCharging();
    if (!isBatteryCharging && stateOfCharge < BUZZER_STATE_OF_CHARGE_LIMIT)
    {
        buzzer.enable();
    }
    else
    {
        buzzer.disable();
    }

    float onboardTemperatureCelsius = onboardTemperature.readCelsius();
    float externalTemperatureCelsius = externalTemperature.readCelsius();
    fan.update(onboardTemperatureCelsius);

    float current = currentVoltageSensor.readCurrent();
    float voltage = currentVoltageSensor.readVoltage();


    DEBUG_SERIAL.print("isBatteryCharging=");
    DEBUG_SERIAL.println(isBatteryCharging);
    DEBUG_SERIAL.print("stateOfCharge=");
    DEBUG_SERIAL.println(stateOfCharge);
    DEBUG_SERIAL.print("current=");
    DEBUG_SERIAL.println(current);
    DEBUG_SERIAL.print("voltage=");
    DEBUG_SERIAL.println(voltage);
    DEBUG_SERIAL.print("onboardTemperatureCelsius=");
    DEBUG_SERIAL.println(onboardTemperatureCelsius);
    DEBUG_SERIAL.print("externalTemperatureCelsius=");
    DEBUG_SERIAL.println(externalTemperatureCelsius);
    DEBUG_SERIAL.println("Send status");
    DEBUG_SERIAL.println();
    opencrCommandSender.sendStatusCommand(isBatteryCharging, stateOfCharge, current, voltage);
}

#endif
