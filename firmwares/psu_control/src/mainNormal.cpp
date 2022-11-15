#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_NORMAL

#include "mainCommon.h"
#include "ShutdownManager.h"

#include <SerialCommunication.h>

#include <Ticker.h>

ShutdownManager shutdownManager(POWER_OFF_PIN, POWER_SWITCH_PIN);

static void setupShutdownManager();

static void onStatusTicker();
static void updateAudioPowerAmplifier(bool isPsuConnected);
static void updateBuzzer(bool isPsuConnected, float stateOfCharge);
static void updateFan(float onboardTemperature, float externalTemperature);
static void updateLedStrip(
    float stateOfCharge,
    uint8_t volume,
    float frontLightSensor,
    float backLightSensor,
    float leftLightSensor,
    float rightLightSensor);

static void onButtonTicker();

static void onSetVolumeMessage(Device source, SetVolumePayload payload);
static void onSetLedColorsMessage(Device source, SetLedColorsPayload payload);

static bool isShutdownCompletedForComputerAndDynamixels();

Ticker statusTicker(onStatusTicker, STATUS_TICKER_INTERVAL_MS, 0, MILLIS);
Ticker buttonTicker(onButtonTicker, BUTTON_TICKER_INTERVAL_MS, 0, MILLIS);

void setup()
{
    initRandom();

    setupDebugSerial();
    setupWire();
    setupPwm();
    setupAdc();

    setupShutdownManager();

    setupAudioPowerAmplifier();
    setupBuzzer();
    setupFan();
    setupLedStrip();

    setupCharger();

    setupLightSensors();
    setupCurrentVoltageSensor();
    setupPushButtons();
    setupThermistors();

    // TODO Register onSetVolumeMessage
    // TODO Register onSetLedCOMessage

    statusTicker.start();
    buttonTicker.start();
}

void loop()
{
    if (shutdownManager.isShutdownRequested())
    {
        // TODO send shutdown message to the computer and dynamixel control.
        shutdownManager.setShutdownRequestHandled();
    }
    else if (shutdownManager.isShutdownPending())
    {
        if (shutdownManager.hasShutdownRequestTimeout() || isShutdownCompletedForComputerAndDynamixels())
        {
            shutdownManager.shutdown();
        }
    }
    else
    {
        statusTicker.update();
        buttonTicker.update();
    }

    // TODO update serial communication
}

static void setupShutdownManager()
{
    DEBUG_SERIAL.println("Setup Shutdown Manager - Start");
    shutdownManager.begin();
    DEBUG_SERIAL.println("Setup Shutdown Manager - End");
}

static void onStatusTicker()
{
    BaseStatusPayload baseStatusPayload;
    baseStatusPayload.isPsuConnected = charger.isPsuConnected();
    baseStatusPayload.hasChargerError = charger.hasChargerError();
    baseStatusPayload.isBatteryCharging = charger.isBatteryCharging();
    baseStatusPayload.hasBatteryError = charger.hasBatteryError();

    if (!battery.readRelativeStateOfCharge(baseStatusPayload.stateOfCharge))
    {
        DEBUG_SERIAL.println("Read relative state of charge failed");
        baseStatusPayload.stateOfCharge = 0.f;
    }

    baseStatusPayload.current = currentVoltageSensor.readCurrent();
    baseStatusPayload.voltage = currentVoltageSensor.readVoltage();
    baseStatusPayload.onboardTemperature = onboardThermistor.readCelsius();
    baseStatusPayload.externalTemperature = externalThermistor.readCelsius();
    baseStatusPayload.frontLightSensor = frontLightSensor.read();
    baseStatusPayload.backLightSensor = backLightSensor.read();
    baseStatusPayload.leftLightSensor = leftLightSensor.read();
    baseStatusPayload.rightLightSensor = rightLightSensor.read();

    updateAudioPowerAmplifier(baseStatusPayload.isPsuConnected);
    baseStatusPayload.volume = audioPowerAmplifier.volume();
    baseStatusPayload.maximumVolume = audioPowerAmplifier.maximumVolume();

    updateBuzzer(baseStatusPayload.isPsuConnected, baseStatusPayload.stateOfCharge);
    updateFan(baseStatusPayload.onboardTemperature, baseStatusPayload.externalTemperature);
    updateLedStrip(
        baseStatusPayload.stateOfCharge,
        baseStatusPayload.volume,
        baseStatusPayload.frontLightSensor,
        baseStatusPayload.backLightSensor,
        baseStatusPayload.leftLightSensor,
        baseStatusPayload.rightLightSensor);

    // TODO send the status
}

static void updateAudioPowerAmplifier(bool isPsuConnected)
{
    if (isPsuConnected)
    {
        audioPowerAmplifier.setMaximumVolume(AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME);
    }
    else
    {
        audioPowerAmplifier.setMaximumVolume(AUDIO_POWER_AMPLIFIER_BATTERY_MAXIMUM_VOLUME);
    }
}

static void updateBuzzer(bool isPsuConnected, float stateOfCharge)
{
    if (!isPsuConnected && stateOfCharge < BUZZER_STATE_OF_CHARGE_LIMIT)
    {
        buzzer.enable();
    }
    else
    {
        buzzer.disable();
    }
}

static void updateFan(float onboardTemperature, float externalTemperature)
{
    fan.update(max(onboardTemperature, externalTemperature));
}

static void updateLedStrip(
    float stateOfCharge,
    uint8_t volume,
    float frontLightSensor,
    float backLightSensor,
    float leftLightSensor,
    float rightLightSensor)
{
    ledStrip.setStateOfCharge(stateOfCharge);
    ledStrip.setVolume(volume);

    float lightLevel = min(min(frontLightSensor, backLightSensor), min(leftLightSensor, rightLightSensor));
    uint8_t brightness = static_cast<uint8_t>(
        LED_STRIP_MINIMUM_BRIGHTNESS + (LED_STRIP_MAXIMUM_BRIGHTNESS - LED_STRIP_MINIMUM_BRIGHTNESS) * lightLevel);
    ledStrip.setBrightness(brightness);
}

static void onButtonTicker()
{
    if (startButton.read())
    {
        // TODO send message
    }
    if (stopButton.read())
    {
        // TODO send message
    }

    bool volumeUpButtonPressed = volumeUpButton.read();
    bool volumeDownButtonPressed = volumeDownButton.read();
    if (volumeUpButtonPressed && !volumeDownButtonPressed && audioPowerAmplifier.volume() < audioPowerAmplifier.maximumVolume())
    {
        audioPowerAmplifier.setVolume(audioPowerAmplifier.volume() + 1);
    }
    if (!volumeUpButtonPressed && volumeDownButtonPressed && audioPowerAmplifier.volume() > 0)
    {
        audioPowerAmplifier.setVolume(audioPowerAmplifier.volume() - 1);
    }
}

static void onSetVolumeMessage(Device source, SetVolumePayload payload)
{
    audioPowerAmplifier.setVolume(payload.volume);
}

static void onSetLedColorsMessage(Device source, SetLedColorsPayload payload)
{
    ledStrip.setBaseLedColors(payload.colors, SetLedColorsPayload::LED_COUNT);
}

static bool isShutdownCompletedForComputerAndDynamixels()
{
    float current = currentVoltageSensor.readCurrent();
    float voltage = currentVoltageSensor.readVoltage();
    float power = current * voltage;

    return power < SHUTDOWN_COMPLETED_FOR_COMPUTER_AND_DYNAMIXELS_POWER_THRESHOLD_W;
}

#endif
