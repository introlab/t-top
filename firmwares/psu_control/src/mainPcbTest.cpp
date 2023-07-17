#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_PCB_TEST

#include "mainCommon.h"

constexpr uint32_t TEST_DELAY_MS = 5000;

#define TEST(function)                                                                                                 \
    function();                                                                                                        \
    DEBUG_SERIAL.println();                                                                                            \
    delay(TEST_DELAY_MS)

void setup()
{
    initRandom();

    setupDebugSerial();
    setupWire();
    setupPwm();
    setupAdc();

    setupAudioPowerAmplifier();
    setupBuzzer();
    setupFan();
    setupLedStrip();

    setupCharger();

    setupLightSensors();
    setupCurrentVoltageSensor();
    setupPushButtons();
    setupThermistors();
}

static void testAudioPowerAmplifier();
static void testBuzzer();
static void testFan();
static void testLedStrip();

static void testBattery();
static void testCharger();

static void testLightSensors();
static void testCurrentVoltageSensor();
static void testPushButtons();
static void testThermistors();

static void testShutdown();

void loop()
{
    TEST(testAudioPowerAmplifier);
    TEST(testBuzzer);
    TEST(testFan);
    TEST(testLedStrip);

    TEST(testBattery);
    TEST(testCharger);

    TEST(testLightSensors);
    TEST(testCurrentVoltageSensor);
    TEST(testPushButtons);
    TEST(testThermistors);

    TEST(testShutdown);
}

static void testAudioPowerAmplifier()
{
    constexpr uint32_t TEST_STEP_DELAY_MS = 2500;

    DEBUG_SERIAL.println("-----------------Test Audio Power Amplifier----------------");

    DEBUG_SERIAL.println("Setting the volume to 50");
    audioPowerAmplifier.setVolume(50);
    DEBUG_SERIAL.print("Actual volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.volume());
    DEBUG_SERIAL.print("Actual maximum volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.maximumVolume());
    DEBUG_SERIAL.println();
    delay(TEST_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Setting the maximum volume to 45");
    audioPowerAmplifier.setMaximumVolume(AUDIO_POWER_AMPLIFIER_BATTERY_MAXIMUM_VOLUME);
    DEBUG_SERIAL.print("Actual volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.volume());
    DEBUG_SERIAL.print("Actual maximum volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.maximumVolume());
    DEBUG_SERIAL.println();
    delay(TEST_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Setting the volume to 24");
    audioPowerAmplifier.setVolume(24);
    DEBUG_SERIAL.print("Actual volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.volume());
    DEBUG_SERIAL.println();
    delay(TEST_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Setting the maximum volume to 63");
    audioPowerAmplifier.setMaximumVolume(AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME);
    DEBUG_SERIAL.print("Actual volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.volume());
    DEBUG_SERIAL.print("Actual maximum volume: ");
    DEBUG_SERIAL.println(audioPowerAmplifier.maximumVolume());
}

static void testBuzzer()
{
    constexpr uint32_t TEST_STEP_DELAY_MS = 5000;

    DEBUG_SERIAL.println("------------------------Test Buzzer------------------------");

    DEBUG_SERIAL.println("Enabling the buzzer");
    buzzer.enable();
    delay(TEST_STEP_DELAY_MS);
    DEBUG_SERIAL.println("Disabling the buzzer");
    buzzer.disable();
}

static void testFan()
{
    constexpr uint32_t TEST_STEP_DELAY_MS = 5000;

    DEBUG_SERIAL.println("--------------------------Test Fan-------------------------");

    DEBUG_SERIAL.println("Half Speed");
    fan.update(FAN_TEMPERATURE_STEP_1 + FAN_HYSTERESIS);
    delay(TEST_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Full Speed");
    fan.update(FAN_TEMPERATURE_STEP_2 + FAN_HYSTERESIS);
    delay(TEST_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Stop");
    fan.update(FAN_TEMPERATURE_STEP_1 - FAN_HYSTERESIS);
}

static void testLedStrip()
{
    constexpr uint32_t TEST_SMALL_STEP_DELAY_MS = 250;
    constexpr uint32_t TEST_BIG_STEP_DELAY_MS = 2500;
    DEBUG_SERIAL.println("---------------------Test Battery LEDs---------------------");

    for (float stateOfCharge = 0.f; stateOfCharge <= 100.f; stateOfCharge += 1)
    {
        DEBUG_SERIAL.print("State of Charge: ");
        DEBUG_SERIAL.println(stateOfCharge);
        ledStrip.setStateOfCharge(stateOfCharge);
        delay(TEST_SMALL_STEP_DELAY_MS);
    }

    DEBUG_SERIAL.println();
    for (uint8_t volume = 0; volume <= AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME; volume++)
    {
        DEBUG_SERIAL.print("Volume: ");
        DEBUG_SERIAL.println(volume);
        ledStrip.setVolume(volume);
        delay(TEST_SMALL_STEP_DELAY_MS);
    }

    Color red[BASE_LED_COUNT];
    Color green[BASE_LED_COUNT];
    Color blue[BASE_LED_COUNT];

    for (size_t i = 0; i < BASE_LED_COUNT; i++)
    {
        red[i] = Color{255, 0, 0};
        green[i] = Color{0, 255, 0};
        blue[i] = Color{0, 0, 255};
    }

    DEBUG_SERIAL.println();
    DEBUG_SERIAL.println("Base LEDs: Red (64)");
    ledStrip.setBrightness(64);
    ledStrip.setBaseLedColors(red, BASE_LED_COUNT);
    delay(TEST_BIG_STEP_DELAY_MS);
    DEBUG_SERIAL.println("Base LEDs: Red (128)");
    ledStrip.setBrightness(128);
    delay(TEST_BIG_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Base LEDs: Green (64)");
    ledStrip.setBrightness(64);
    ledStrip.setBaseLedColors(green, BASE_LED_COUNT);
    delay(TEST_BIG_STEP_DELAY_MS);
    DEBUG_SERIAL.println("Base LEDs: Green (128)");
    ledStrip.setBrightness(128);
    delay(TEST_BIG_STEP_DELAY_MS);

    DEBUG_SERIAL.println("Base LEDs: Blue (64)");
    ledStrip.setBrightness(64);
    ledStrip.setBaseLedColors(blue, BASE_LED_COUNT);
    delay(TEST_BIG_STEP_DELAY_MS);
    DEBUG_SERIAL.println("Base LEDs: Blue (128)");
    ledStrip.setBrightness(128);
}


static void testBattery()
{
    constexpr uint8_t MAX_NAME_SIZE = 128;
    char name[MAX_NAME_SIZE];
    int day, month, year;
    uint16_t serialNumber;
    uint16_t cycleCount;

    float temperature;
    float voltage;
    float current;
    float capacity;
    float stateOfCharge;
    float time;
    bool isFullyDischarged, isFullyCharged;
    RrcBatteryErrorCode error;

    DEBUG_SERIAL.println("------------------------Test Battery-----------------------");
    if (battery.readManufacturerName(name, MAX_NAME_SIZE))
    {
        DEBUG_SERIAL.print("Manufacturer Name: ");
        DEBUG_SERIAL.println(name);
    }
    if (battery.readDeviceName(name, MAX_NAME_SIZE))
    {
        DEBUG_SERIAL.print("Device Name: ");
        DEBUG_SERIAL.println(name);
    }
    if (battery.readDeviceChemistry(name, MAX_NAME_SIZE))
    {
        DEBUG_SERIAL.print("Device Chemistry: ");
        DEBUG_SERIAL.println(name);
    }
    if (battery.readManufactureDate(day, month, year))
    {
        DEBUG_SERIAL.print("Manufacture Date (YYYY/MM/DD): ");
        DEBUG_SERIAL.print(year);
        DEBUG_SERIAL.print("/");
        DEBUG_SERIAL.print(month);
        DEBUG_SERIAL.print("/");
        DEBUG_SERIAL.println(day);
    }
    if (battery.readSerialNumber(serialNumber))
    {
        DEBUG_SERIAL.print("Serial Number: ");
        DEBUG_SERIAL.println(serialNumber);
    }

    Serial.println();
    if (battery.readDesignVoltage(voltage))
    {
        DEBUG_SERIAL.print("Design Voltage: ");
        DEBUG_SERIAL.print(voltage);
        DEBUG_SERIAL.println(" V");
    }
    if (battery.readDesignCapacity(capacity))
    {
        DEBUG_SERIAL.print("Design Capacity: ");
        DEBUG_SERIAL.print(capacity);
        DEBUG_SERIAL.println(" Ah");
    }

    DEBUG_SERIAL.println();
    if (battery.readBatteryStatus(isFullyDischarged, isFullyCharged, error))
    {
        DEBUG_SERIAL.print("Battey Status: isFullyDischarged=");
        DEBUG_SERIAL.print(isFullyDischarged);
        DEBUG_SERIAL.print(", isFullyCharged=");
        DEBUG_SERIAL.print(isFullyCharged);
        DEBUG_SERIAL.print(", error=");
        DEBUG_SERIAL.println(static_cast<int>(error));
    }
    if (battery.readCycleCount(cycleCount))
    {
        DEBUG_SERIAL.print("Cycle Count: ");
        DEBUG_SERIAL.println(cycleCount);
    }
    if (battery.readTemperature(temperature))
    {
        DEBUG_SERIAL.print("Temperature: ");
        DEBUG_SERIAL.print(temperature);
        DEBUG_SERIAL.println(" C");
    }
    if (battery.readVoltage(voltage))
    {
        DEBUG_SERIAL.print("Voltage: ");
        DEBUG_SERIAL.print(voltage);
        DEBUG_SERIAL.println(" V");
    }
    if (battery.readCurrent(current))
    {
        DEBUG_SERIAL.print("Current: ");
        DEBUG_SERIAL.print(current);
        DEBUG_SERIAL.println(" A");
    }
    if (battery.readAverageCurrent(current))
    {
        DEBUG_SERIAL.print("Average Current: ");
        DEBUG_SERIAL.print(current);
        DEBUG_SERIAL.println(" A");
    }
    if (battery.readRelativeStateOfCharge(stateOfCharge))
    {
        DEBUG_SERIAL.print("Relative State of Charge: ");
        DEBUG_SERIAL.print(stateOfCharge);
        DEBUG_SERIAL.println("%");
    }
    if (battery.readAbsoluteStateOfCharge(stateOfCharge))
    {
        DEBUG_SERIAL.print("Absolute State of Charge: ");
        DEBUG_SERIAL.print(stateOfCharge);
        DEBUG_SERIAL.println("%");
    }
    if (battery.readRemainingCapacity(capacity))
    {
        DEBUG_SERIAL.print("Remaining Capacity: ");
        DEBUG_SERIAL.print(capacity);
        DEBUG_SERIAL.println(" Ah");
    }
    if (battery.readFullChargeCapacity(capacity))
    {
        DEBUG_SERIAL.print("Full Charge Capacity: ");
        DEBUG_SERIAL.print(capacity);
        DEBUG_SERIAL.println(" Ah");
    }
    if (battery.readFullChargeCapacity(capacity))
    {
        DEBUG_SERIAL.print("Full Charge Capacity: ");
        DEBUG_SERIAL.print(capacity);
        DEBUG_SERIAL.println(" Ah");
    }
    if (battery.readRunTimeToEmpty(time))
    {
        DEBUG_SERIAL.print("Runtime To Empty: ");
        DEBUG_SERIAL.print(time);
        DEBUG_SERIAL.println(" min");
    }
    if (battery.readAverageTimeToEmpty(time))
    {
        DEBUG_SERIAL.print("Average Runtime To Empty: ");
        DEBUG_SERIAL.print(time);
        DEBUG_SERIAL.println(" min");
    }
    if (battery.readAverageTimeToFull(time))
    {
        DEBUG_SERIAL.print("Average Runtime To Full: ");
        DEBUG_SERIAL.print(time);
        DEBUG_SERIAL.println(" min");
    }
}

static void testCharger()
{
    DEBUG_SERIAL.println("------------------------Test Charger-----------------------");

    DEBUG_SERIAL.print("isBatteryCharged=");
    DEBUG_SERIAL.println(charger.isBatteryCharged());

    DEBUG_SERIAL.print("isBatteryCharging=");
    DEBUG_SERIAL.println(charger.isBatteryCharging());

    DEBUG_SERIAL.print("hasBatteryError=");
    DEBUG_SERIAL.println(charger.hasBatteryError());

    DEBUG_SERIAL.print("isPsuConnected=");
    DEBUG_SERIAL.println(charger.isPsuConnected());
    DEBUG_SERIAL.print("hasChargerError=");
    DEBUG_SERIAL.println(charger.hasChargerError());
}

static void testLightSensor(const char* name, AlsPt19LightSensor& lightSensor)
{
    constexpr size_t READ_COUNT = 20;
    constexpr uint32_t TEST_STEP_DELAY_MS = 1000;

    DEBUG_SERIAL.print(name);
    DEBUG_SERIAL.println(":");
    for (size_t i = 0; i < READ_COUNT; i++)
    {
        DEBUG_SERIAL.println(lightSensor.read());
        delay(TEST_STEP_DELAY_MS);
    }
    DEBUG_SERIAL.println();
}

static void testLightSensors()
{
    DEBUG_SERIAL.println("---------------------Test Light Sensors--------------------");

    testLightSensor("Front", frontLightSensor);
    testLightSensor("Back", backLightSensor);
    testLightSensor("Left", leftLightSensor);
    testLightSensor("Right", rightLightSensor);
}

static void testCurrentVoltageSensor()
{
    DEBUG_SERIAL.println("----------------Test Current/Voltage Sensor----------------");

    DEBUG_SERIAL.print("Current");
    DEBUG_SERIAL.print(currentVoltageSensor.readCurrent());
    DEBUG_SERIAL.println(" A");

    DEBUG_SERIAL.print("Voltage");
    DEBUG_SERIAL.print(currentVoltageSensor.readVoltage());
    DEBUG_SERIAL.println(" V");
}

static void testPushButton(const char* name, PushButton& button)
{
    constexpr size_t READ_COUNT = 20;
    constexpr uint32_t TEST_STEP_DELAY_MS = 500;

    DEBUG_SERIAL.print(name);
    DEBUG_SERIAL.println(":");
    for (size_t i = 0; i < READ_COUNT; i++)
    {
        DEBUG_SERIAL.println(button.read());
        delay(TEST_STEP_DELAY_MS);
    }
    DEBUG_SERIAL.println();
}

static void testPushButtons()
{
    DEBUG_SERIAL.println("---------------------Test Push Buttons---------------------");

    testPushButton("Start", startButton);
    testPushButton("Stop", stopButton);
    testPushButton("Volume Up", volumeUpButton);
    testPushButton("Volume Down", volumeDownButton);
}

static void testThermistors()
{
    DEBUG_SERIAL.println("----------------------Test Thermistors---------------------");
    DEBUG_SERIAL.print("Onboard Thermistors: ");
    DEBUG_SERIAL.print(onboardThermistor.readCelsius());
    DEBUG_SERIAL.println(" C");

    DEBUG_SERIAL.print("External Thermistors: ");
    DEBUG_SERIAL.print(externalThermistor.readCelsius());
    DEBUG_SERIAL.println(" C");
}

static void testShutdown()
{
    constexpr size_t READ_COUNT = 20;
    constexpr uint32_t TEST_STEP_DELAY_MS = 1000;
    constexpr uint32_t SHUTDOWN_DELAY_MS = 1000;

    DEBUG_SERIAL.println("---------------------Test Shutdown---------------------");

    pinMode(POWER_OFF_PIN, OUTPUT);
    digitalWrite(POWER_OFF_PIN, true);
    pinMode(POWER_SWITCH_PIN, INPUT);

    DEBUG_SERIAL.println("Power switch pin :");
    for (size_t i = 0; i < READ_COUNT; i++)
    {
        DEBUG_SERIAL.println(static_cast<int>(digitalRead(POWER_SWITCH_PIN)));
        delay(TEST_STEP_DELAY_MS);
    }
    DEBUG_SERIAL.println();

    DEBUG_SERIAL.println("Shutdown");
    digitalWrite(POWER_OFF_PIN, false);
    delay(SHUTDOWN_DELAY_MS);
}

#endif
