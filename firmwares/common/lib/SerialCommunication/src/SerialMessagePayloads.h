#ifndef SERIAL_COMMUNICATION_SERIAL_MESSAGE_PAYLOADS_H
#define SERIAL_COMMUNICATION_SERIAL_MESSAGE_PAYLOADS_H

#include "SerialCommunicationBuffer.h"
#include "SerialMessages.h"


struct AcknowledgmentPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = false;
    static constexpr MessageType MESSAGE_TYPE = MessageType::ACKNOWLEDGMENT;
    static constexpr uint8_t PAYLOAD_SIZE = 2;

    uint16_t receivedMessageId;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<AcknowledgmentPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool AcknowledgmentPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(receivedMessageId));

    return true;
}

template<class Buffer>
tl::optional<AcknowledgmentPayload> AcknowledgmentPayload::readFrom(Buffer& buffer)
{
    AcknowledgmentPayload payload;
    CHECK_BUFFER_READ(payload.receivedMessageId, buffer.template read<uint16_t>());

    return payload;
}


struct BaseStatusPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = false;
    static constexpr MessageType MESSAGE_TYPE = MessageType::BASE_STATUS;
    static constexpr uint8_t PAYLOAD_SIZE = 42;

    bool isPsuConnected;
    bool hasChargerError;
    bool isBatteryCharging;
    bool hasBatteryError;
    float stateOfCharge;
    float current;
    float voltage;
    float onboardTemperature;
    float externalTemperature;
    float frontLightSensor;
    float backLightSensor;
    float leftLightSensor;
    float rightLightSensor;
    uint8_t volume;
    uint8_t maximumVolume;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<BaseStatusPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool BaseStatusPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(isPsuConnected));
    CHECK_BUFFER_WRITE(buffer.write(hasChargerError));
    CHECK_BUFFER_WRITE(buffer.write(isBatteryCharging));
    CHECK_BUFFER_WRITE(buffer.write(hasBatteryError));
    CHECK_BUFFER_WRITE(buffer.write(stateOfCharge));
    CHECK_BUFFER_WRITE(buffer.write(current));
    CHECK_BUFFER_WRITE(buffer.write(voltage));
    CHECK_BUFFER_WRITE(buffer.write(onboardTemperature));
    CHECK_BUFFER_WRITE(buffer.write(externalTemperature));
    CHECK_BUFFER_WRITE(buffer.write(frontLightSensor));
    CHECK_BUFFER_WRITE(buffer.write(backLightSensor));
    CHECK_BUFFER_WRITE(buffer.write(leftLightSensor));
    CHECK_BUFFER_WRITE(buffer.write(rightLightSensor));
    CHECK_BUFFER_WRITE(buffer.write(volume));
    CHECK_BUFFER_WRITE(buffer.write(maximumVolume));

    return true;
}

template<class Buffer>
tl::optional<BaseStatusPayload> BaseStatusPayload::readFrom(Buffer& buffer)
{
    BaseStatusPayload payload;
    CHECK_BUFFER_READ(payload.isPsuConnected, buffer.template read<bool>());
    CHECK_BUFFER_READ(payload.hasChargerError, buffer.template read<bool>());
    CHECK_BUFFER_READ(payload.isBatteryCharging, buffer.template read<bool>());
    CHECK_BUFFER_READ(payload.hasBatteryError, buffer.template read<bool>());
    CHECK_BUFFER_READ(payload.stateOfCharge, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.current, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.voltage, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.onboardTemperature, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.externalTemperature, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.frontLightSensor, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.backLightSensor, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.leftLightSensor, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.rightLightSensor, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.volume, buffer.template read<uint8_t>());
    CHECK_BUFFER_READ(payload.maximumVolume, buffer.template read<uint8_t>());

    return payload;
}


enum class Button : uint8_t
{
    START = 0,
    STOP = 1
};

template<>
struct enum_min<Button> : std::integral_constant<uint8_t, 0>
{
};

template<>
struct enum_max<Button> : std::integral_constant<uint8_t, 1>
{
};

struct ButtonPressedPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::BUTTON_PRESSED;
    static constexpr uint8_t PAYLOAD_SIZE = 1;

    Button button;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<ButtonPressedPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool ButtonPressedPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(button));

    return true;
}

template<class Buffer>
tl::optional<ButtonPressedPayload> ButtonPressedPayload::readFrom(Buffer& buffer)
{
    ButtonPressedPayload payload;
    CHECK_BUFFER_READ(payload.button, buffer.template read<Button>());

    return payload;
}


struct SetVolumePayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::SET_VOLUME;
    static constexpr uint8_t PAYLOAD_SIZE = 1;

    uint8_t volume;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<SetVolumePayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool SetVolumePayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(volume));

    return true;
}

template<class Buffer>
tl::optional<SetVolumePayload> SetVolumePayload::readFrom(Buffer& buffer)
{
    SetVolumePayload payload;
    CHECK_BUFFER_READ(payload.volume, buffer.template read<uint8_t>());

    return payload;
}


struct Color
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct SetLedColorsPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::SET_LED_COLORS;
    static constexpr uint8_t PAYLOAD_SIZE = 84;

    static constexpr size_t LED_COUNT = 28;
    Color colors[LED_COUNT];

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<SetLedColorsPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool SetLedColorsPayload::writeTo(Buffer& buffer) const
{
    for (size_t i = 0; i < LED_COUNT; i++)
    {
        CHECK_BUFFER_WRITE(buffer.write(colors[i].red));
        CHECK_BUFFER_WRITE(buffer.write(colors[i].green));
        CHECK_BUFFER_WRITE(buffer.write(colors[i].blue));
    }

    return true;
}

template<class Buffer>
tl::optional<SetLedColorsPayload> SetLedColorsPayload::readFrom(Buffer& buffer)
{
    SetLedColorsPayload payload;

    for (size_t i = 0; i < LED_COUNT; i++)
    {
        CHECK_BUFFER_READ(payload.colors[i].red, buffer.template read<uint8_t>());
        CHECK_BUFFER_READ(payload.colors[i].green, buffer.template read<uint8_t>());
        CHECK_BUFFER_READ(payload.colors[i].blue, buffer.template read<uint8_t>());
    }

    return payload;
}


struct MotorStatusPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = false;
    static constexpr MessageType MESSAGE_TYPE = MessageType::MOTOR_STATUS;
    static constexpr uint8_t PAYLOAD_SIZE = 71;

    float torsoOrientation;
    int16_t torsoServoSpeed;
    float headServoAngle1;
    float headServoAngle2;
    float headServoAngle3;
    float headServoAngle4;
    float headServoAngle5;
    float headServoAngle6;
    int16_t headServoSpeed1;
    int16_t headServoSpeed2;
    int16_t headServoSpeed3;
    int16_t headServoSpeed4;
    int16_t headServoSpeed5;
    int16_t headServoSpeed6;
    float headPosePositionX;
    float headPosePositionY;
    float headPosePositionZ;
    float headPoseOrientationW;
    float headPoseOrientationX;
    float headPoseOrientationY;
    float headPoseOrientationZ;
    bool isHeadPoseReachable;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<MotorStatusPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool MotorStatusPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(torsoOrientation));
    CHECK_BUFFER_WRITE(buffer.write(torsoServoSpeed));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle1));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle2));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle3));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle4));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle5));
    CHECK_BUFFER_WRITE(buffer.write(headServoAngle6));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed1));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed2));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed3));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed4));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed5));
    CHECK_BUFFER_WRITE(buffer.write(headServoSpeed6));
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionX));
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionY));
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionZ));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationW));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationX));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationY));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationZ));
    CHECK_BUFFER_WRITE(buffer.write(isHeadPoseReachable));

    return true;
}

template<class Buffer>
tl::optional<MotorStatusPayload> MotorStatusPayload::readFrom(Buffer& buffer)
{
    MotorStatusPayload payload;
    CHECK_BUFFER_READ(payload.torsoOrientation, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.torsoServoSpeed, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoAngle1, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoAngle2, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoAngle3, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoAngle4, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoAngle5, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoAngle6, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headServoSpeed1, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoSpeed2, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoSpeed3, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoSpeed4, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoSpeed5, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headServoSpeed6, buffer.template read<int16_t>());
    CHECK_BUFFER_READ(payload.headPosePositionX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPosePositionY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPosePositionZ, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationW, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationZ, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.isHeadPoseReachable, buffer.template read<bool>());

    return payload;
}


struct ImuDataPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = false;
    static constexpr MessageType MESSAGE_TYPE = MessageType::IMU_DATA;
    static constexpr uint8_t PAYLOAD_SIZE = 24;

    float accelerationX;
    float accelerationY;
    float accelerationZ;
    float angularRateX;
    float angularRateY;
    float angularRateZ;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<ImuDataPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool ImuDataPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(accelerationX));
    CHECK_BUFFER_WRITE(buffer.write(accelerationY));
    CHECK_BUFFER_WRITE(buffer.write(accelerationZ));
    CHECK_BUFFER_WRITE(buffer.write(angularRateX));
    CHECK_BUFFER_WRITE(buffer.write(angularRateY));
    CHECK_BUFFER_WRITE(buffer.write(angularRateZ));

    return true;
}

template<class Buffer>
tl::optional<ImuDataPayload> ImuDataPayload::readFrom(Buffer& buffer)
{
    ImuDataPayload payload;
    CHECK_BUFFER_READ(payload.accelerationX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.accelerationY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.accelerationZ, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.angularRateX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.angularRateY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.angularRateZ, buffer.template read<float>());

    return payload;
}


struct SetTorsoOrientationPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::SET_TORSO_ORIENTATION;
    static constexpr uint8_t PAYLOAD_SIZE = 4;

    float torsoOrientation;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<SetTorsoOrientationPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool SetTorsoOrientationPayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(torsoOrientation));

    return true;
}

template<class Buffer>
tl::optional<SetTorsoOrientationPayload> SetTorsoOrientationPayload::readFrom(Buffer& buffer)
{
    SetTorsoOrientationPayload payload;
    CHECK_BUFFER_READ(payload.torsoOrientation, buffer.template read<float>());

    return payload;
}


struct SetHeadPosePayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::SET_HEAD_POSE;
    static constexpr uint8_t PAYLOAD_SIZE = 28;

    float headPosePositionX;
    float headPosePositionY;
    float headPosePositionZ;
    float headPoseOrientationW;
    float headPoseOrientationX;
    float headPoseOrientationY;
    float headPoseOrientationZ;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<SetHeadPosePayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool SetHeadPosePayload::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionX));
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionY));
    CHECK_BUFFER_WRITE(buffer.write(headPosePositionZ));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationW));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationX));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationY));
    CHECK_BUFFER_WRITE(buffer.write(headPoseOrientationZ));

    return true;
}

template<class Buffer>
tl::optional<SetHeadPosePayload> SetHeadPosePayload::readFrom(Buffer& buffer)
{
    SetHeadPosePayload payload;
    CHECK_BUFFER_READ(payload.headPosePositionX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPosePositionY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPosePositionZ, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationW, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationX, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationY, buffer.template read<float>());
    CHECK_BUFFER_READ(payload.headPoseOrientationZ, buffer.template read<float>());

    return payload;
}


struct ShutdownPayload
{
    static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = true;
    static constexpr MessageType MESSAGE_TYPE = MessageType::SHUTDOWN;
    static constexpr uint8_t PAYLOAD_SIZE = 0;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<ShutdownPayload> readFrom(Buffer& buffer);
};

template<class Buffer>
bool ShutdownPayload::writeTo(Buffer& buffer) const
{
    return true;
}

template<class Buffer>
tl::optional<ShutdownPayload> ShutdownPayload::readFrom(Buffer& buffer)
{
    return ShutdownPayload();
}

#endif
