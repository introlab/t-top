#include "SerialMessagePayloads.h"

constexpr bool AcknowledgmentPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType AcknowledgmentPayload::MESSAGE_TYPE;

constexpr bool BaseStatusPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType BaseStatusPayload::MESSAGE_TYPE;

constexpr bool ButtonPressedPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ButtonPressedPayload::MESSAGE_TYPE;

constexpr bool SetVolumePayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetVolumePayload::MESSAGE_TYPE;

constexpr bool SetLEDColorsPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetLEDColorsPayload::MESSAGE_TYPE;
constexpr size_t SetLEDColorsPayload::LED_COUNT;

constexpr bool MotorStatusPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType MotorStatusPayload::MESSAGE_TYPE;

constexpr bool ImuDataPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ImuDataPayload::MESSAGE_TYPE;

constexpr bool SetTorsoOrientationPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetTorsoOrientationPayload::MESSAGE_TYPE;

constexpr bool SetHeadPosePayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetHeadPosePayload::MESSAGE_TYPE;

constexpr bool ShutdownPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ShutdownPayload::MESSAGE_TYPE;
