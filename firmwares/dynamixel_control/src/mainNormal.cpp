#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_NORMAL

#include "mainCommon.h"

#include <SerialCommunication.h>

#include <Ticker.h>

static TeensySerialPort<decltype(COMPUTER_COMMUNICATION_SERIAL)>
    computerSerialCommunicationSerialPort(COMPUTER_COMMUNICATION_SERIAL);
static SerialCommunicationManager computerSerialCommunicationManager(
    Device::DYNAMIXEL_CONTROL,
    COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
    COMMUNICATION_MAXIMUM_TRIAL_COUNT,
    computerSerialCommunicationSerialPort);

static TeensySerialPort<decltype(PSU_CONTROL_COMMUNICATION_SERIAL)>
    psuControlSerialCommunicationSerialPort(PSU_CONTROL_COMMUNICATION_SERIAL);
static SerialCommunicationManager psuControlSerialCommunicationManager(
    Device::DYNAMIXEL_CONTROL,
    COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
    COMMUNICATION_MAXIMUM_TRIAL_COUNT,
    psuControlSerialCommunicationSerialPort);

static void setupSerialCommunicationManagers();

static void onMotorStatusTicker();
static void onImuTicker();

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload);
static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload);
static void onShutdownMessage(Device source, const ShutdownPayload& payload);
static void computerRouteCallback(Device destination, const uint8_t* data, size_t size);
static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size);
static void logInvalidRoute(const char* device, Device destination);
static void onComputerSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onPsuControlSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);

Ticker motorStatusTicker(onMotorStatusTicker, MOTOR_STATUS_TICKER_INTERVAL_MS, 0, MILLIS);
Ticker imuTicker(onImuTicker, IMU_TICKER_INTERVAL_MS, 0, MILLIS);

void setup()
{
    // put your setup code here, to run once:
    setupDebugSerial();
    setupWire();

    setupSerialCommunicationManagers();

    motorStatusTicker.start();
    imuTicker.start();
}

void loop()
{
    motorStatusTicker.update();
    imuTicker.update();

    computerSerialCommunicationManager.update(millis());
    psuControlSerialCommunicationManager.update(millis());
}

static void setupSerialCommunicationManagers()
{
    DEBUG_SERIAL.println("Setup Serial Communication Manager - Start");

    COMPUTER_COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD_RATE);
    computerSerialCommunicationManager.setSetHeadPoseHandler(onSetHeadPoseMessage);
    computerSerialCommunicationManager.setSetTorsoOrientationHandler(onSetTorsoOrientationMessage);
    computerSerialCommunicationManager.setRouteCallback(computerRouteCallback);
    computerSerialCommunicationManager.setErrorCallback(onComputerSerialCommunicationError);

    pinMode(PSU_CONTROL_COMMUNICATION_RS232_INVALID, INPUT);
    PSU_CONTROL_COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD_RATE);
    psuControlSerialCommunicationManager.setShutdownHandler(onShutdownMessage);
    psuControlSerialCommunicationManager.setRouteCallback(psuControlRouteCallback);
    psuControlSerialCommunicationManager.setErrorCallback(onPsuControlSerialCommunicationError);

    DEBUG_SERIAL.println("Setup Serial Communication Manager - End");
}

static void onMotorStatusTicker()
{
    MotorStatusPayload motorStatusPayload;
    // TODO set fields

    computerSerialCommunicationManager.send(Device::COMPUTER, motorStatusPayload, millis());
}

static void onImuTicker()
{
    ImuDataPayload imuDataPayload;
    // TODO set fields

    computerSerialCommunicationManager.send(Device::COMPUTER, imuDataPayload, millis());
}

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload)
{
    // TODO
}

static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload)
{
    // TODO
}

static void onShutdownMessage(Device source, const ShutdownPayload& payload)
{
    // TODO
}

static void computerRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::PSU_CONTROL)
    {
        logInvalidRoute("Computer", destination);
        return;
    }

    psuControlSerialCommunicationManager.sendRaw(data, size);
}

static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::COMPUTER)
    {
        logInvalidRoute("PSU Control", destination);
        return;
    }

    computerSerialCommunicationManager.sendRaw(data, size);
}

static void logInvalidRoute(const char* device, Device destination)
{
    DEBUG_SERIAL.print(device);
    DEBUG_SERIAL.print("Invalid Route - destination=");
    DEBUG_SERIAL.print(static_cast<int>(destination));
    DEBUG_SERIAL.println();
}

static void onComputerSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("Computer ");
    onSerialCommunicationError(message, messageType);
}

static void onPsuControlSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("PSU Control ");
    onSerialCommunicationError(message, messageType);
}

static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("Serial Communication Manager - ");
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