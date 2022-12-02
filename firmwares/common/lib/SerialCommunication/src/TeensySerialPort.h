#ifndef PSU_CONTROL_TEENSY_SERIAL_PORT_H
#define PSU_CONTROL_TEENSY_SERIAL_PORT_H

#include "SerialCommunicationManager.h"

template<class T>
class TeensySerialPort : public SerialPort
{
    T& m_serial;

public:
    TeensySerialPort(T& serial);

    void read(SerialCommunicationBufferView& buffer) override;
    void write(const uint8_t* data, size_t size) override;
};

template<class T>
TeensySerialPort<T>::TeensySerialPort(T& serial) : m_serial(serial)
{
}

template<class T>
void TeensySerialPort<T>::read(SerialCommunicationBufferView& buffer)
{
    while (buffer.sizeToWrite() > 0 && m_serial.available() > 0)
    {
        buffer.write(static_cast<uint8_t>(m_serial.read()));
    }
}

template<class T>
void TeensySerialPort<T>::write(const uint8_t* data, size_t size)
{
    m_serial.write(data, size);
}

#endif
