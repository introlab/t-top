#include "DaemonSerialPortWrapper.h"
#include <QDebug>
#include <QByteArray>

DaemonSerialPortWrapper::DaemonSerialPortWrapper(const QSerialPortInfo &info, QObject *parent)
    : QSerialPort(info, parent)
{

}

void DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)
{
    //qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " asking for: " << buffer.sizeToWrite();
    //qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " available: " << this->bytesAvailable();
    size_t read_size = std::min(buffer.sizeToWrite(), (size_t) this->bytesAvailable());
    //qDebug() << "DaemonSerialPortWrapper::read(SerialCommunicationBufferView &buffer)" << " read_size: " << read_size;
    QByteArray data = QSerialPort::read(read_size);

    if (data.size() == read_size)
    {
        buffer.write(reinterpret_cast<const uint8_t*>(data.constData()), data.size());
    }
    else
    {
        qDebug() << "Reading buffer... error expected: " << read_size <<" got: "<<data.size();
    }
}

void DaemonSerialPortWrapper::write(const uint8_t *data, size_t size)
{
    // qDebug() << " DaemonSerialPortWrapper::write(const uint8_t *data, size_t size)" << size;
    size_t write_size  = QSerialPort::write(reinterpret_cast<const char*>(data), size);
    if (write_size != size)
    {
        qDebug() << "Writing buffer... error expected: " << size <<" got: "<< write_size;
    }
}
