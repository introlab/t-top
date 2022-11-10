#ifndef BATTERY_LED_H
#define BATTERY_LED_H

class BatteryLed
{
public:
    BatteryLed();
    void begin();

    void setStateOfCharge(float stateOfCharge);
};

#endif
