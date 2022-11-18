#include "DaemonSerialPortWrapper.h"
#include <QDebug>

DaemonSerialPortWrapper::DaemonSerialPortWrapper(const QSerialPortInfo &info, QObject *parent)
    : QSerialPort(info, parent)
{

}

void DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)
{
    qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " asking for: " << buffer.sizeToWrite();
    qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " available: " << this->bytesAvailable();
    size_t read_size = std::min(buffer.sizeToWrite(), (size_t) this->bytesAvailable());

    qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " read_size: " << read_size;

    uint8_t data[read_size];
    this->readData(reinterpret_cast<char*>(data), read_size);
    buffer.write(data, read_size);
}

void DaemonSerialPortWrapper::write(const uint8_t *data, size_t size)
{
    qDebug() << " DaemonSerialPortWrapper::write(const uint8_t *data, size_t size)" << size;
    this->write(data, size);
}
