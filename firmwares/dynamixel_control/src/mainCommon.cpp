#include "mainCommon.h"

void setupDebugSerial()
{
    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD_RATE);
}

void setupWire()
{
    DEBUG_SERIAL.println("Setup Wire - Start");
    Wire.setClock(WIRE_CLOCK);
    Wire.begin();
    DEBUG_SERIAL.println("Setup Wire - End");
}