#include "SerialMessagePayloads.h"

constexpr bool AcknowledgmentPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType AcknowledgmentPayload::MESSAGE_TYPE;
constexpr uint8_t AcknowledgmentPayload::PAYLOAD_SIZE;

constexpr bool BaseStatusPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType BaseStatusPayload::MESSAGE_TYPE;
constexpr uint8_t BaseStatusPayload::PAYLOAD_SIZE;

constexpr bool ButtonPressedPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ButtonPressedPayload::MESSAGE_TYPE;
constexpr uint8_t ButtonPressedPayload::PAYLOAD_SIZE;

constexpr bool SetVolumePayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetVolumePayload::MESSAGE_TYPE;
constexpr uint8_t SetVolumePayload::PAYLOAD_SIZE;

constexpr bool SetLedColorsPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetLedColorsPayload::MESSAGE_TYPE;
constexpr size_t SetLedColorsPayload::LED_COUNT;
constexpr uint8_t SetLedColorsPayload::PAYLOAD_SIZE;

constexpr bool MotorStatusPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType MotorStatusPayload::MESSAGE_TYPE;
constexpr uint8_t MotorStatusPayload::PAYLOAD_SIZE;

constexpr bool ImuDataPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ImuDataPayload::MESSAGE_TYPE;
constexpr uint8_t ImuDataPayload::PAYLOAD_SIZE;

constexpr bool SetTorsoOrientationPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetTorsoOrientationPayload::MESSAGE_TYPE;
constexpr uint8_t SetTorsoOrientationPayload::PAYLOAD_SIZE;

constexpr bool SetHeadPosePayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType SetHeadPosePayload::MESSAGE_TYPE;
constexpr uint8_t SetHeadPosePayload::PAYLOAD_SIZE;

constexpr bool ShutdownPayload::DEFAULT_ACKNOWLEDGMENT_NEEDED;
constexpr MessageType ShutdownPayload::MESSAGE_TYPE;
constexpr uint8_t ShutdownPayload::PAYLOAD_SIZE;
