#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_COMMUNICATION_TEST

#include "mainCommon.h"

#include <SerialCommunication.h>

static TeensySerialPort<decltype(COMMUNICATION_SERIAL)> serialCommunicationSerialPort(COMMUNICATION_SERIAL);
static SerialCommunicationManager serialCommunicationManager(
    Device::PSU_CONTROL,
    COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
    COMMUNICATION_MAXIMUM_TRIAL_COUNT,
    serialCommunicationSerialPort);

static void setupSerialCommunicationManager();

static void onSetVolumeMessage(Device source, const SetVolumePayload& payload, void* userData);
static void onSetLedColorsMessage(Device source, const SetLedColorsPayload& payload, void* userData);
static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType, void* userData);

void setup()
{
    initRandom();

    setupDebugSerial();
    setupSerialCommunicationManager();
}

static void setupSerialCommunicationManager()
{
    pinMode(COMMUNICATION_RS232_INVALID, INPUT);
    COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD_RATE);
    serialCommunicationManager.setSetVolumeHandler(onSetVolumeMessage);
    serialCommunicationManager.setSetLedColorsHandler(onSetLedColorsMessage);
    serialCommunicationManager.setErrorCallback(onSerialCommunicationError);
}

template<class Payload>
static void sendMessage(const char* name, Device destination, Payload payload)
{
    uint32_t startTime = micros();
    serialCommunicationManager.send(destination, payload, millis());
    uint32_t endTime = micros();

    DEBUG_SERIAL.print("Elapsed time for sending ");
    DEBUG_SERIAL.print(name);
    DEBUG_SERIAL.print(": ");
    DEBUG_SERIAL.print(endTime - startTime);
    DEBUG_SERIAL.println(" us");

    serialCommunicationManager.update(millis());
}

void loop()
{
    sendMessage("shutdown", Device::COMPUTER, ShutdownPayload());
    sendMessage("shutdown", Device::DYNAMIXEL_CONTROL, ShutdownPayload());
    sendMessage("button pressed", Device::COMPUTER, ButtonPressedPayload{Button::START});
    sendMessage("button pressed", Device::COMPUTER, ButtonPressedPayload{Button::STOP});

    BaseStatusPayload baseStatusPayload;
    baseStatusPayload.isPsuConnected = true;
    baseStatusPayload.hasChargerError = false;
    baseStatusPayload.isBatteryCharging = true;
    baseStatusPayload.hasBatteryError = false;
    baseStatusPayload.stateOfCharge = 55;
    baseStatusPayload.current = 1.1;
    baseStatusPayload.voltage = 19.1;
    baseStatusPayload.onboardTemperature = 26.1;
    baseStatusPayload.externalTemperature = 29.2;
    baseStatusPayload.frontLightSensor = 0.25;
    baseStatusPayload.backLightSensor = 0.26;
    baseStatusPayload.leftLightSensor = 0.27;
    baseStatusPayload.rightLightSensor = 0.28;
    sendMessage("base status", Device::COMPUTER, baseStatusPayload);

    uint32_t startTime = millis();
    while ((millis() - startTime) < 1000)
    {
        serialCommunicationManager.update(millis());
    }
}

static void onSetVolumeMessage(Device source, const SetVolumePayload& payload, void* userData)
{
    DEBUG_SERIAL.print("Serial Communication Manager Set Volume - ");
    DEBUG_SERIAL.print(static_cast<int>(payload.volume));
    DEBUG_SERIAL.print(" (source=");
    DEBUG_SERIAL.print(static_cast<int>(source));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();
}

static void onSetLedColorsMessage(Device source, const SetLedColorsPayload& payload, void* userData)
{
    DEBUG_SERIAL.print("Serial Communication Manager Set Led Colors - ");
    DEBUG_SERIAL.print(" (source=");
    DEBUG_SERIAL.print(static_cast<int>(source));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();
}

static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType, void* userData)
{
    DEBUG_SERIAL.print("Serial Communication Manager Error - ");
    DEBUG_SERIAL.print(message);
    if (messageType.has_value())
    {
        DEBUG_SERIAL.print(" (messageType=");
        DEBUG_SERIAL.print(static_cast<int>(*messageType));
        DEBUG_SERIAL.print(")");
    }
    DEBUG_SERIAL.println();
}

#endif
