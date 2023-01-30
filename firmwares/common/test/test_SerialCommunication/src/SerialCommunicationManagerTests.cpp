#include <SerialCommunication.h>

#include <gtest/gtest.h>

#include <vector>

using namespace std;

struct CheckPendingMessagesSetFunctor
{
    template<class Payload>
    void operator()(tl::optional<PendingMessage<Payload>>& message)
    {
        EXPECT_EQ(message, tl::nullopt);
    }

    void operator()(tl::optional<PendingMessage<ButtonPressedPayload>>& message)
    {
        EXPECT_TRUE(message.has_value());
        EXPECT_EQ(message->timestampMs, 10);
        EXPECT_EQ(message->trialCount, 1);
        EXPECT_EQ(message->message.header().source(), Device::COMPUTER);
        EXPECT_EQ(message->message.header().destination(), Device::DYNAMIXEL_CONTROL);
        EXPECT_EQ(message->message.header().messageId(), 150);
        EXPECT_EQ(message->message.payload().button, Button::STOP);
    }
};


TEST(PendingMessagesTests, setClear_shouldSetClearTheMessage)
{
    MessageHeader::setMessageIdCounter(150);

    PendingMessages testee;
    testee.forEach([](auto& message) { EXPECT_EQ(message, tl::nullopt); });

    Message<ButtonPressedPayload> message(
        Device::COMPUTER,
        Device::DYNAMIXEL_CONTROL,
        ButtonPressedPayload{Button::STOP});
    testee.set(message, 10);
    testee.forEach(CheckPendingMessagesSetFunctor());
    testee.clear(140);
    testee.forEach(CheckPendingMessagesSetFunctor());

    testee.clear(150);
    testee.forEach([](auto& message) { EXPECT_EQ(message, tl::nullopt); });
}

class SerialPortMock : public SerialPort
{
    size_t m_rxDataIndex;
    vector<uint8_t> m_rxData;
    vector<uint8_t> m_txData;

public:
    SerialPortMock() : m_rxDataIndex(0) {}

    void writeRxData(uint8_t data) { m_rxData.push_back(data); }

    void writeRxData(const uint8_t* data, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            m_rxData.push_back(data[i]);
        }
    }

    vector<uint8_t>& txData() { return m_txData; }

    void read(SerialCommunicationBufferView& buffer) override
    {
        while (m_rxData.size() > m_rxDataIndex)
        {
            buffer.write(m_rxData[m_rxDataIndex]);
            m_rxDataIndex++;
        }
    }

    void write(const uint8_t* data, size_t size) override
    {
        for (size_t i = 0; i < size; i++)
        {
            m_txData.push_back(data[i]);
        }
    }
};

constexpr uint32_t ACKNOWLEDGMENT_TIMEOUT_MS = 10;
constexpr size_t MAXIMUM_TRIAL_COUNT = 3;

#define EXPECT_NO_CALLBACK()                                                                                           \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_BASE_STATUS()                                                                        \
    EXPECT_NE(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED()                                                                     \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_NE(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_SET_VOLUME()                                                                         \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_NE(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_SET_LED_COLORS()                                                                     \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_NE(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_MOTOR_STATUS()                                                                       \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_NE(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_IMU_DATA()                                                                           \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_NE(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_SET_TORSO_ORIENTATION()                                                              \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_NE(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_SET_HEAD_POSE()                                                                      \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_NE(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_CALLBACK_EXCEPT_SHUTDOWN()                                                                           \
    EXPECT_EQ(receivedBaseStatusPayloads.size(), 0);                                                                   \
    EXPECT_EQ(receivedButtonPressedPayloads.size(), 0);                                                                \
    EXPECT_EQ(receivedSetVolumePayloads.size(), 0);                                                                    \
    EXPECT_EQ(receivedSetLedColorsPayloads.size(), 0);                                                                 \
    EXPECT_EQ(receivedMotorStatusPayloads.size(), 0);                                                                  \
    EXPECT_EQ(receivedImuDataPayloads.size(), 0);                                                                      \
    EXPECT_EQ(receivedSetTorsoOrientationPayloads.size(), 0);                                                          \
    EXPECT_EQ(receivedSetHeadPosePayloads.size(), 0);                                                                  \
    EXPECT_NE(receivedShutdownPayloads.size(), 0)

#define EXPECT_NO_ROUTE_CALLBACK()                                                                                     \
    EXPECT_EQ(routedDataDestination, tl::nullopt);                                                                     \
    EXPECT_EQ(routedData.size(), 0)


class SerialCommunicationManagerTests : public ::testing::Test
{
protected:
    SerialPortMock serialPortMock;
    SerialCommunicationManager testee;

    vector<BaseStatusPayload> receivedBaseStatusPayloads;
    vector<ButtonPressedPayload> receivedButtonPressedPayloads;
    vector<SetVolumePayload> receivedSetVolumePayloads;
    vector<SetLedColorsPayload> receivedSetLedColorsPayloads;
    vector<MotorStatusPayload> receivedMotorStatusPayloads;
    vector<ImuDataPayload> receivedImuDataPayloads;
    vector<SetTorsoOrientationPayload> receivedSetTorsoOrientationPayloads;
    vector<SetHeadPosePayload> receivedSetHeadPosePayloads;
    vector<ShutdownPayload> receivedShutdownPayloads;

    tl::optional<Device> routedDataDestination;
    vector<uint8_t> routedData;
    tl::optional<const char*> expectedErrorMessage;

    SerialCommunicationManagerTests()
        : testee(Device::PSU_CONTROL, ACKNOWLEDGMENT_TIMEOUT_MS, MAXIMUM_TRIAL_COUNT, serialPortMock)
    {
    }

    void SetUp() override
    {
        testee.setBaseStatusHandler([this](Device source, const BaseStatusPayload& payload)
                                    { baseStatusHandler(source, payload); });
        testee.setButtonPressedHandler([this](Device source, const ButtonPressedPayload& payload)
                                       { buttonPressedHandler(source, payload); });
        testee.setSetVolumeHandler([this](Device source, const SetVolumePayload& payload)
                                   { setVolumeHandler(source, payload); });
        testee.setSetLedColorsHandler([this](Device source, const SetLedColorsPayload& payload)
                                      { setLedColorsHandler(source, payload); });
        testee.setMotorStatusHandler([this](Device source, const MotorStatusPayload& payload)
                                     { motorStatusHandler(source, payload); });
        testee.setImuDataHandler([this](Device source, const ImuDataPayload& payload)
                                 { imuDataHandler(source, payload); });
        testee.setSetTorsoOrientationHandler([this](Device source, const SetTorsoOrientationPayload& payload)
                                             { setTorsoOrientationHandler(source, payload); });
        testee.setSetHeadPoseHandler([this](Device source, const SetHeadPosePayload& payload)
                                     { setHeadPoseHandler(source, payload); });
        testee.setShutdownHandler([this](Device source, const ShutdownPayload& payload)
                                  { shutdownHandler(source, payload); });
        testee.setRouteCallback([this](Device destination, const uint8_t* data, size_t size)
                                { routeCallback(destination, data, size); });
        testee.setErrorCallback([this](const char* message, tl::optional<MessageType> messageType)
                                { errorCallback(message, messageType); });
    }

    void TearDown() override
    {
        if (expectedErrorMessage != tl::nullopt)
        {
            ADD_FAILURE() << *expectedErrorMessage;
        }
    }

    void baseStatusHandler(Device source, const BaseStatusPayload& payload)
    {
        receivedBaseStatusPayloads.push_back(payload);
    }

    void buttonPressedHandler(Device source, const ButtonPressedPayload& payload)
    {
        receivedButtonPressedPayloads.push_back(payload);
    }

    void setVolumeHandler(Device source, const SetVolumePayload& payload)
    {
        receivedSetVolumePayloads.push_back(payload);
    }

    void setLedColorsHandler(Device source, const SetLedColorsPayload& payload)
    {
        receivedSetLedColorsPayloads.push_back(payload);
    }

    void motorStatusHandler(Device source, const MotorStatusPayload& payload)
    {
        receivedMotorStatusPayloads.push_back(payload);
    }

    void imuDataHandler(Device source, const ImuDataPayload& payload) { receivedImuDataPayloads.push_back(payload); }

    void setTorsoOrientationHandler(Device source, const SetTorsoOrientationPayload payload)
    {
        receivedSetTorsoOrientationPayloads.push_back(payload);
    }

    void setHeadPoseHandler(Device source, const SetHeadPosePayload& payload)
    {
        receivedSetHeadPosePayloads.push_back(payload);
    }

    void shutdownHandler(Device source, const ShutdownPayload& payload) { receivedShutdownPayloads.push_back(payload); }

    void routeCallback(Device destination, const uint8_t* data, size_t size)
    {
        routedDataDestination = destination;
        for (size_t i = 0; i < size; i++)
        {
            routedData.push_back(data[i]);
        }
    }

    void errorCallback(const char* message, tl::optional<MessageType> messageType)
    {
        if (expectedErrorMessage == tl::nullopt)
        {
            ADD_FAILURE() << message;
        }
        else
        {
            EXPECT_STREQ(*expectedErrorMessage, message);
            expectedErrorMessage = tl::nullopt;
        }
    }
};

TEST_F(SerialCommunicationManagerTests, readBytePerByte_shouldCallTheSpecificCallback)
{
    serialPortMock.writeRxData(0x01);  // Junk
    testee.update(1);
    serialPortMock.writeRxData(0x02);  // Junk
    testee.update(2);
    serialPortMock.writeRxData(0x03);  // Junk
    testee.update(3);
    serialPortMock.writeRxData(0x04);  // Junk
    testee.update(4);
    EXPECT_NO_CALLBACK();

    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(5);
    serialPortMock.writeRxData(0x02);  // Junk
    testee.update(6);
    serialPortMock.writeRxData(0x03);  // Junk
    testee.update(7);
    serialPortMock.writeRxData(0x04);  // Junk
    testee.update(8);
    EXPECT_NO_CALLBACK();

    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(9);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(10);
    serialPortMock.writeRxData(0x03);  // Junk
    testee.update(11);
    serialPortMock.writeRxData(0x04);  // Junk
    testee.update(12);
    EXPECT_NO_CALLBACK();

    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(13);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(14);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(15);
    serialPortMock.writeRxData(0x04);  // Junk
    testee.update(16);
    EXPECT_NO_CALLBACK();

    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(17);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(18);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(19);
    serialPortMock.writeRxData(0xAA);  // Preamble byte
    testee.update(20);
    EXPECT_NO_CALLBACK();

    serialPortMock.writeRxData(10);  // Message length
    testee.update(21);
    serialPortMock.writeRxData(static_cast<uint8_t>(Device::COMPUTER));  // Source
    testee.update(22);
    serialPortMock.writeRxData(static_cast<uint8_t>(Device::PSU_CONTROL));  // Destination
    testee.update(23);
    serialPortMock.writeRxData(static_cast<uint8_t>(false));  // Acknowledgment needed
    testee.update(24);
    serialPortMock.writeRxData(0x01);  // Message id
    serialPortMock.writeRxData(0x02);
    testee.update(25);
    serialPortMock.writeRxData(0x02);  // Message Type (button pressed)
    serialPortMock.writeRxData(0x00);
    testee.update(26);
    serialPortMock.writeRxData(static_cast<uint8_t>(Button::STOP));  // Payload
    testee.update(23);
    serialPortMock.writeRxData(0x8A);  // CRC8
    testee.update(23);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED();
    ASSERT_EQ(receivedButtonPressedPayloads.size(), 1);
    EXPECT_EQ(receivedButtonPressedPayloads[0].button, Button::STOP);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_baseStatus_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x32,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0x01,
        0x02,
        0x01,
        0x00,  // Header

        // Payload
        0x01,
        0x00,
        0x01,
        0x00,  // isPsuConnected, hasChargerError, isBatteryCharging, hasBatteryError
        0x00,
        0x00,
        0x80,
        0x3f,  // stateOfCharge
        0x00,
        0x00,
        0x00,
        0x40,  // current
        0x00,
        0x00,
        0x40,
        0x40,  // voltage
        0x00,
        0x00,
        0x80,
        0x40,  // onboardTemperature
        0x00,
        0x00,
        0xA0,
        0x40,  // externalTemperature
        0x00,
        0x00,
        0xC0,
        0x40,  // frontLightSensor
        0x00,
        0x00,
        0xE0,
        0x40,  // backLightSensor
        0x00,
        0x00,
        0x10,
        0x41,  // leftLightSensor
        0x00,
        0x00,
        0x20,
        0x41,  // rightLightSensor
        0x11,
        0x22,  // volume, maximumVolume

        0x07  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BASE_STATUS();
    ASSERT_EQ(receivedBaseStatusPayloads.size(), 1);
    EXPECT_TRUE(receivedBaseStatusPayloads[0].isPsuConnected);
    EXPECT_FALSE(receivedBaseStatusPayloads[0].hasChargerError);
    EXPECT_TRUE(receivedBaseStatusPayloads[0].isBatteryCharging);
    EXPECT_FALSE(receivedBaseStatusPayloads[0].hasBatteryError);
    EXPECT_EQ(receivedBaseStatusPayloads[0].stateOfCharge, 1.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].current, 2.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].voltage, 3.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].onboardTemperature, 4.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].externalTemperature, 5.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].frontLightSensor, 6.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].backLightSensor, 7.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].leftLightSensor, 9.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].rightLightSensor, 10.f);
    EXPECT_EQ(receivedBaseStatusPayloads[0].volume, 0x11);
    EXPECT_EQ(receivedBaseStatusPayloads[0].maximumVolume, 0x22);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_buttonPressed_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0x01,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x8A  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED();
    ASSERT_EQ(receivedButtonPressedPayloads.size(), 1);
    EXPECT_EQ(receivedButtonPressedPayloads[0].button, Button::STOP);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_setVolume_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x03,
        0x00,  // Header
        0x25,  // Payload
        0x6F  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_SET_VOLUME();
    ASSERT_EQ(receivedSetVolumePayloads.size(), 1);
    EXPECT_EQ(receivedSetVolumePayloads[0].volume, 37);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_setLedColors_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x5D,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x04,
        0x00,  // Header
        // Payload
        1,
        2,
        3,
        2,
        3,
        4,
        3,
        4,
        5,
        4,
        5,
        6,
        5,
        6,
        7,
        6,
        7,
        8,
        7,
        8,
        9,
        8,
        9,
        10,
        9,
        10,
        11,
        10,
        11,
        12,
        11,
        12,
        13,
        12,
        13,
        14,
        13,
        14,
        15,
        14,
        15,
        16,
        15,
        16,
        17,
        16,
        17,
        18,
        17,
        18,
        19,
        18,
        19,
        20,
        19,
        20,
        21,
        20,
        21,
        22,
        21,
        22,
        23,
        22,
        23,
        24,
        23,
        24,
        25,
        24,
        25,
        26,
        25,
        26,
        27,
        26,
        27,
        28,
        27,
        28,
        29,
        28,
        29,
        30,
        0x8C  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_SET_LED_COLORS();
    ASSERT_EQ(receivedSetLedColorsPayloads.size(), 1);
    for (size_t i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        EXPECT_EQ(receivedSetLedColorsPayloads[0].colors[i].red, i + 1);
        EXPECT_EQ(receivedSetLedColorsPayloads[0].colors[i].green, i + 2);
        EXPECT_EQ(receivedSetLedColorsPayloads[0].colors[i].blue, i + 3);
    }
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_motorStatus_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x50,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x05,
        0x00,  // Header

        // Payload
        0x00,
        0x00,
        0x80,
        0x3F,  // torsoOrientation
        0x02,
        0x00,  // torsoServoSpeed
        0x00,
        0x00,
        0x40,
        0x40,  // headServoAngle1
        0x00,
        0x00,
        0x80,
        0x40,  // headServoAngle2
        0x00,
        0x00,
        0xA0,
        0x40,  // headServoAngle3
        0x00,
        0x00,
        0xC0,
        0x40,  // headServoAngle4
        0x00,
        0x00,
        0xE0,
        0x40,  // headServoAngle5
        0x00,
        0x00,
        0x00,
        0x41,  // headServoAngle6
        0x09,
        0x00,
        0x0A,
        0x00,  // headServoSpeed1, headServoSpeed2
        0x0B,
        0x00,
        0x0C,
        0x00,  // headServoSpeed3, headServoSpeed4
        0x0D,
        0x00,
        0x0E,
        0x00,  // headServoSpeed5, headServoSpeed6
        0x00,
        0x00,
        0x70,
        0x41,  // headPosePositionX
        0x00,
        0x00,
        0x80,
        0x41,  // headPosePositionY
        0x00,
        0x00,
        0x88,
        0x41,  // headPosePositionZ
        0x00,
        0x00,
        0x90,
        0x41,  // headPoseOrientationW
        0x00,
        0x00,
        0x98,
        0x41,  // headPoseOrientationX
        0x00,
        0x00,
        0xA0,
        0x41,  // headPoseOrientationY
        0x00,
        0x00,
        0xA8,
        0x41,  // headPoseOrientationZ
        0x01,  // isHeadPoseReachable

        0xE6  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_MOTOR_STATUS();
    ASSERT_EQ(receivedMotorStatusPayloads.size(), 1);
    EXPECT_EQ(receivedMotorStatusPayloads[0].torsoOrientation, 1.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].torsoServoSpeed, 2);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle1, 3.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle2, 4.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle3, 5.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle4, 6.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle5, 7.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoAngle6, 8.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed1, 9);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed2, 10);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed3, 11);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed4, 12);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed5, 13);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headServoSpeed6, 14);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPosePositionX, 15.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPosePositionY, 16.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPosePositionZ, 17.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPoseOrientationW, 18.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPoseOrientationX, 19.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPoseOrientationY, 20.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].headPoseOrientationZ, 21.f);
    EXPECT_EQ(receivedMotorStatusPayloads[0].isHeadPoseReachable, true);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_2Messages_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0x01,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x8A  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);
    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED();
    ASSERT_EQ(receivedButtonPressedPayloads.size(), 2);
    EXPECT_EQ(receivedButtonPressedPayloads[0].button, Button::STOP);
    EXPECT_EQ(receivedButtonPressedPayloads[1].button, Button::STOP);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_imuData_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x21,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x06,
        0x00,  // Header

        // Payload
        0x00,
        0x00,
        0x80,
        0x3F,  // accelerationX
        0x00,
        0x00,
        0x00,
        0x40,  // accelerationY
        0x00,
        0x00,
        0x40,
        0x40,  // accelerationZ
        0x00,
        0x00,
        0x80,
        0x40,  // angularRateX
        0x00,
        0x00,
        0xA0,
        0x40,  // angularRateY
        0x00,
        0x00,
        0xC0,
        0x40,  // angularRateZ

        0x40  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_IMU_DATA();
    ASSERT_EQ(receivedImuDataPayloads.size(), 1);
    EXPECT_EQ(receivedImuDataPayloads[0].accelerationX, 1.f);
    EXPECT_EQ(receivedImuDataPayloads[0].accelerationY, 2);
    EXPECT_EQ(receivedImuDataPayloads[0].accelerationZ, 3.f);
    EXPECT_EQ(receivedImuDataPayloads[0].angularRateX, 4.f);
    EXPECT_EQ(receivedImuDataPayloads[0].angularRateY, 5.f);
    EXPECT_EQ(receivedImuDataPayloads[0].angularRateZ, 6.f);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_setTorsoOrientation_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0D,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x07,
        0x00,  // Header

        // Payload
        0x00,
        0x00,
        0x80,
        0x3F,  // torsoOrientation

        0x39  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_CALLBACK_EXCEPT_SET_TORSO_ORIENTATION();
    ASSERT_EQ(receivedSetTorsoOrientationPayloads.size(), 1);
    EXPECT_EQ(receivedSetTorsoOrientationPayloads[0].torsoOrientation, 1.f);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_setHeadPose_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x25,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x08,
        0x00,  // Header

        // Payload
        0x00,
        0x00,
        0x80,
        0x3F,  // headPosePositionX
        0x00,
        0x00,
        0x00,
        0x40,  // headPosePositionY
        0x00,
        0x00,
        0x40,
        0x40,  // headPosePositionZ
        0x00,
        0x00,
        0x80,
        0x40,  // headPoseOrientationW
        0x00,
        0x00,
        0xA0,
        0x40,  // headPoseOrientationX
        0x00,
        0x00,
        0xC0,
        0x40,  // headPoseOrientationY
        0x00,
        0x00,
        0xE0,
        0x40,  // headPoseOrientationZ

        0xA2  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_SET_HEAD_POSE();
    ASSERT_EQ(receivedSetHeadPosePayloads.size(), 1);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPosePositionX, 1.f);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPosePositionY, 2);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPosePositionZ, 3.f);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPoseOrientationW, 4.f);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPoseOrientationX, 5.f);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPoseOrientationY, 6.f);
    EXPECT_EQ(receivedSetHeadPosePayloads[0].headPoseOrientationZ, 7.f);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_shutdown_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x09,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0xAA,
        0xBB,
        0x09,
        0x00,  // Header
        // No payload
        0xEA  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_SHUTDOWN();
    ASSERT_EQ(receivedShutdownPayloads.size(), 1);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_2MessagesAtTheSameTime_shouldCallTheSpecificCallback)
{
    constexpr uint8_t DATA[] = {
        0x01,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x8A,  // CRC8

        0x01,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0x02,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0x03,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0x04,  // Junk
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x00,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::START),  // Payload
        0x8D  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED();
    ASSERT_EQ(receivedButtonPressedPayloads.size(), 2);
    EXPECT_EQ(receivedButtonPressedPayloads[0].button, Button::STOP);
    EXPECT_EQ(receivedButtonPressedPayloads[1].button, Button::START);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_acknowledgmentNeeded_shouldSendAcknowledgment)
{
    MessageHeader::setMessageIdCounter(0x020B);

    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0xA3,  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK_EXCEPT_BUTTON_PRESSED();
    ASSERT_EQ(receivedButtonPressedPayloads.size(), 1);
    EXPECT_EQ(receivedButtonPressedPayloads[0].button, Button::STOP);

    const vector<uint8_t> EXPECTED_TX_DATA = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0B,
        static_cast<uint8_t>(Device::PSU_CONTROL),
        static_cast<uint8_t>(Device::COMPUTER),  // Header
        0x00,
        0x0B,
        0x02,
        0x00,
        0x00,  // Header
        0x01,
        0x02,
        0x3C};
    EXPECT_EQ(serialPortMock.txData(), EXPECTED_TX_DATA);
}

TEST_F(SerialCommunicationManagerTests, read_acknowledgmentNeededWrongCrc_shouldLogAnError)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x00,  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    expectedErrorMessage = "CRC8 Error";
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, read_acknowledgmentNeededNotHandledMessage_shouldLogAnError)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0xA3,  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    expectedErrorMessage = "Not handled message";
    testee.setButtonPressedHandler(nullptr);
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();
    EXPECT_EQ(serialPortMock.txData().size(), 15);
}

TEST_F(SerialCommunicationManagerTests, read_wrongDestination_shouldCallRouteCallback)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::DYNAMIXEL_CONTROL),  // Header
        0x01,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x7C,  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    testee.update(1);

    EXPECT_NO_CALLBACK();
    EXPECT_EQ(routedDataDestination, Device::DYNAMIXEL_CONTROL);
    EXPECT_EQ(routedData, std::vector<uint8_t>(DATA + 4, DATA + sizeof(DATA)));
}

TEST_F(SerialCommunicationManagerTests, read_wrongDestinationNoRouteCallback_shouldLogAnError)
{
    constexpr uint8_t DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,  // Preamble
        0x0A,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::DYNAMIXEL_CONTROL),  // Header
        0x01,
        0x01,
        0x02,
        0x02,
        0x00,  // Header
        static_cast<uint8_t>(Button::STOP),  // Payload
        0x7C,  // CRC8
    };

    serialPortMock.writeRxData(DATA, sizeof(DATA));
    expectedErrorMessage = "No Route Callback: The message is dropped.";
    testee.setRouteCallback(nullptr);
    testee.update(1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();
}

TEST_F(SerialCommunicationManagerTests, send_shouldSendTheMessage)
{
    MessageHeader::setMessageIdCounter(0x020B);
    ButtonPressedPayload payload{Button::STOP};
    testee.send(Device::COMPUTER, false, payload, 1);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();

    const vector<uint8_t> EXPECTED_TX_DATA = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0A,
        static_cast<uint8_t>(Device::PSU_CONTROL),
        static_cast<uint8_t>(Device::COMPUTER),  // Header
        0x00,
        0x0B,
        0x02,
        0x02,
        0x00,  // Header
        0x01,  // Payload
        0xC8  // CRC8
    };
    EXPECT_EQ(serialPortMock.txData(), EXPECTED_TX_DATA);
}

TEST_F(SerialCommunicationManagerTests, send_acknowledgmentNeeded_shouldSendTheMessageUntilAcknowledgment)
{
    MessageHeader::setMessageIdCounter(0x020B);
    ButtonPressedPayload payload{Button::STOP};
    testee.send(Device::COMPUTER, payload, 0);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();

    const vector<uint8_t> EXPECTED_TX_DATA = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0A,
        static_cast<uint8_t>(Device::PSU_CONTROL),
        static_cast<uint8_t>(Device::COMPUTER),  // Header
        0x01,
        0x0B,
        0x02,
        0x02,
        0x00,  // Header
        0x01,  // Payload
        0xE1  // CRC8
    };
    EXPECT_EQ(serialPortMock.txData(), EXPECTED_TX_DATA);

    // Don't resend the until the timeout is reached.
    serialPortMock.txData().clear();
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS / 2);
    EXPECT_EQ(serialPortMock.txData().size(), 0);

    // Resend the message after the timeout.
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData(), EXPECTED_TX_DATA);

    constexpr uint8_t ACKNOWLEDGMENT_DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0B,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x0B,
        0x02,
        0x00,
        0x00,  // Header
        0x0B,
        0x02,  // Payload
        0xB5  // CRC8
    };
    serialPortMock.writeRxData(ACKNOWLEDGMENT_DATA, sizeof(ACKNOWLEDGMENT_DATA));
    serialPortMock.txData().clear();

    testee.update(2 * ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, send_acknowledgmentNeededInvalidSource_shouldSendTheMessageNTimes)
{
    MessageHeader::setMessageIdCounter(0x020B);
    ButtonPressedPayload payload{Button::STOP};
    testee.send(Device::COMPUTER, payload, 0);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    // Don't resend the until the timeout is reached.
    serialPortMock.txData().clear();
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS / 2);
    EXPECT_EQ(serialPortMock.txData().size(), 0);

    // Resend the message after the timeout.
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    constexpr uint8_t ACKNOWLEDGMENT_DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0B,
        static_cast<uint8_t>(Device::DYNAMIXEL_CONTROL),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x0B,
        0x02,
        0x00,
        0x00,  // Header
        0x0B,
        0x02,  // Payload
        0x3E  // CRC8
    };
    serialPortMock.writeRxData(ACKNOWLEDGMENT_DATA, sizeof(ACKNOWLEDGMENT_DATA));

    serialPortMock.txData().clear();
    testee.update(2 * ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    serialPortMock.txData().clear();
    testee.update(2 * ACKNOWLEDGMENT_TIMEOUT_MS + ACKNOWLEDGMENT_TIMEOUT_MS / 2);
    EXPECT_EQ(serialPortMock.txData().size(), 0);

    expectedErrorMessage = "Too many trials: The message is dropped.";
    serialPortMock.txData().clear();
    testee.update(3 * ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, send_acknowledgmentNeededInvalidCrc_shouldSendTheMessageNTimes)
{
    MessageHeader::setMessageIdCounter(0x020B);
    ButtonPressedPayload payload{Button::STOP};
    testee.send(Device::COMPUTER, payload, 0);

    EXPECT_NO_ROUTE_CALLBACK();
    EXPECT_NO_CALLBACK();
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    // Don't resend the until the timeout is reached.
    serialPortMock.txData().clear();
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS / 2);
    EXPECT_EQ(serialPortMock.txData().size(), 0);

    // Resend the message after the timeout.
    testee.update(ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    constexpr uint8_t ACKNOWLEDGMENT_DATA[] = {
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0x0B,
        static_cast<uint8_t>(Device::COMPUTER),
        static_cast<uint8_t>(Device::PSU_CONTROL),  // Header
        0x01,
        0x0B,
        0x02,
        0x00,
        0x00,  // Header
        0x0B,
        0x02,  // Payload
        0x00  // CRC8
    };
    serialPortMock.writeRxData(ACKNOWLEDGMENT_DATA, sizeof(ACKNOWLEDGMENT_DATA));

    expectedErrorMessage = "CRC8 error";
    serialPortMock.txData().clear();
    testee.update(2 * ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 14);

    serialPortMock.txData().clear();
    testee.update(2 * ACKNOWLEDGMENT_TIMEOUT_MS + ACKNOWLEDGMENT_TIMEOUT_MS / 2);
    EXPECT_EQ(serialPortMock.txData().size(), 0);

    expectedErrorMessage = "Too many trials: The message is dropped.";
    serialPortMock.txData().clear();
    testee.update(3 * ACKNOWLEDGMENT_TIMEOUT_MS);
    EXPECT_EQ(serialPortMock.txData().size(), 0);
}

TEST_F(SerialCommunicationManagerTests, sendRaw_shouldSendThePreambleAndTheData)
{
    constexpr uint8_t DATA = 0x11;
    testee.sendRaw(&DATA, sizeof(DATA));

    const vector<uint8_t> EXPECTED_TX_DATA = {0xAA, 0xAA, 0xAA, 0xAA, 0x11};
    EXPECT_EQ(serialPortMock.txData(), EXPECTED_TX_DATA);
}
