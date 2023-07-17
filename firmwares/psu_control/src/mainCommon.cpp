#include "mainCommon.h"

AudioPowerAmplifier
    audioPowerAmplifier(AUDIO_POWER_AMPLIFIER_WIRE, AUDIO_POWER_AMPLIFIER_I2C_ADDRESSES, AUDIO_POWER_AMPLIFIER_COUNT);
Buzzer buzzer(BUZZER_PIN);
Fan fan(FAN_PIN);
LedStrip ledStrip(LED_STRIP_PIN);

Rrc20542Battery battery(BATTERY_WIRE);
RrcPmm240Charger charger(BATTERY_WIRE, BATTERY_STATUS_PIN, CHARGER_STATUS_PIN);

AlsPt19LightSensor frontLightSensor(FRONT_LIGHT_SENSOR_PIN);
AlsPt19LightSensor backLightSensor(BACK_LIGHT_SENSOR_PIN);
AlsPt19LightSensor leftLightSensor(LEFT_LIGHT_SENSOR_PIN);
AlsPt19LightSensor rightLightSensor(RIGHT_LIGHT_SENSOR_PIN);
CurrentVoltageSensor currentVoltageSensor(
    CURRENT_VOLTAGE_SENSOR_WIRE,
    CURRENT_VOLTAGE_SENSOR_ADDRESS,
    CURRENT_VOLTAGE_SENSOR_SHUNT_RESISTOR,
    CURRENT_VOLTAGE_SENSOR_MAX_CURRENT);
PushButton startButton(START_BUTTON_PIN, PushButtonType::SINGLE);
PushButton stopButton(STOP_BUTTON_PIN, PushButtonType::SINGLE);
PushButton volumeUpButton(VOLUME_UP_BUTTON_PIN, PushButtonType::REPEATABLE);
PushButton volumeDownButton(VOLUME_DOWN_BUTTON_PIN, PushButtonType::REPEATABLE);
Thermistor onboardThermistor(
    ONBOARD_TEMPERATURE_PIN,
    ONBOARD_TEMPERATURE_NTC_R,
    ONBOARD_TEMPERATURE_NTC_BETA,
    ONBOARD_TEMPERATURE_R);
Thermistor externalThermistor(
    EXTERNAL_TEMPERATURE_PIN,
    EXTERNAL_TEMPERATURE_NTC_R,
    EXTERNAL_TEMPERATURE_NTC_BETA,
    EXTERNAL_TEMPERATURE_R);

void setupDebugSerial()
{
    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);
}

void setupWire()
{
    DEBUG_SERIAL.println("Setup Wire - Start");
    Wire.setClock(WIRE_CLOCK);
    Wire1.setClock(WIRE_CLOCK);
    Wire.begin();
    Wire1.begin();
    DEBUG_SERIAL.println("Setup Wire - End");
}

void setupPwm()
{
    DEBUG_SERIAL.println("Setup PWM - Start");
    analogWriteResolution(PWM_RESOLUTION);
    DEBUG_SERIAL.println("Setup PWM - End");
}

void setupAdc()
{
    DEBUG_SERIAL.println("Setup ADC - Start");
    analogReadResolution(ADC_RESOLUTION);
    DEBUG_SERIAL.println("Setup ADC - End");
}


void setupAudioPowerAmplifier()
{
    DEBUG_SERIAL.println("Setup Audio Power Amplifier - Start");
    audioPowerAmplifier.begin();
    DEBUG_SERIAL.println("Setup Audio Power Amplifier - End");
}

void setupBuzzer()
{
    DEBUG_SERIAL.println("Setup Buzzer - Start");
    if (!buzzer.begin())
    {
        CRITICAL_ERROR("Setup Buzzer - failure");
    }
    DEBUG_SERIAL.println("Setup Buzzer - End");
}

void setupFan()
{
    DEBUG_SERIAL.println("Setup Fan - Start");
    fan.begin();
    DEBUG_SERIAL.println("Setup Fan - End");
}

void setupLedStrip()
{
    DEBUG_SERIAL.println("Setup LED Strip - Start");
    if (!ledStrip.begin())
    {
        CRITICAL_ERROR("Setup LED Strip - failure");
    }
    DEBUG_SERIAL.println("Setup LED Strip - End");
}

void setupCharger()
{
    DEBUG_SERIAL.println("Setup Charger - Start");
    charger.begin();
    DEBUG_SERIAL.println("Setup Charger - End");
}

void setupLightSensors()
{
    DEBUG_SERIAL.println("Setup Light Sensors - Start");
    frontLightSensor.begin();
    backLightSensor.begin();
    leftLightSensor.begin();
    rightLightSensor.begin();
    DEBUG_SERIAL.println("Setup Light Sensors - End");
}

void setupCurrentVoltageSensor()
{
    DEBUG_SERIAL.println("Setup Current Voltage Sensor - Start");
    if (!currentVoltageSensor.begin())
    {
        DEBUG_SERIAL.println("Setup Current Voltage Sensor - Failure");
    }
    DEBUG_SERIAL.println("Setup Current Voltage Sensor - End");
}

void setupPushButtons()
{
    DEBUG_SERIAL.println("Setup Push Buttons - Start");
    if (!startButton.begin())
    {
        DEBUG_SERIAL.println("Setup Push Buttons - Failure (start)");
    }
    if (!stopButton.begin())
    {
        DEBUG_SERIAL.println("Setup Push Buttons - Failure (stop)");
    }
    if (!volumeUpButton.begin())
    {
        DEBUG_SERIAL.println("Setup Push Buttons - Failure (volume up)");
    }
    if (!volumeDownButton.begin())
    {
        DEBUG_SERIAL.println("Setup Push Buttons - Failure (volume down)");
    }
    DEBUG_SERIAL.println("Setup Push Buttons - End");
}

void setupThermistors()
{
    DEBUG_SERIAL.println("Setup Thermistors - Start");
    onboardThermistor.begin();
    externalThermistor.begin();
    DEBUG_SERIAL.println("Setup Thermistors - End");
}
