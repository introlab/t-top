#include "DaemonSerialPortWrapper.h"
#include <QDebug>
#include <QByteArray>

DaemonSerialPortWrapper::DaemonSerialPortWrapper(const QSerialPortInfo& info, QObject* parent)
    : QObject(parent),
      m_serialPort(info, this)
{
    // Signal on signal...
    connect(&m_serialPort, &QSerialPort::errorOccurred, this, &DaemonSerialPortWrapper::errorOccurred);
    connect(&m_serialPort, &QSerialPort::readyRead, this, &DaemonSerialPortWrapper::readyRead);


    // TODO port setup, hardcoded for now
    m_serialPort.setDataBits(QSerialPort::Data8);
    m_serialPort.setStopBits(QSerialPort::OneStop);
    m_serialPort.setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
    m_serialPort.setFlowControl(QSerialPort::NoFlowControl);
}

void DaemonSerialPortWrapper::read(SerialCommunicationBufferView& buffer)
{
    size_t read_size = std::min(buffer.sizeToWrite(), (size_t)m_serialPort.bytesAvailable());
    QByteArray data = m_serialPort.read(read_size);

    if (data.size() == read_size)
    {
        buffer.write(reinterpret_cast<const uint8_t*>(data.constData()), data.size());
    }
    else
    {
        qDebug() << "Reading buffer... error expected: " << read_size << " got: " << data.size();
    }
}

void DaemonSerialPortWrapper::write(const uint8_t* data, size_t size)
{
    size_t write_size = m_serialPort.write(reinterpret_cast<const char*>(data), size);
    m_serialPort.flush();
    if (write_size != size)
    {
        qDebug() << "Writing buffer... error expected: " << size << " got: " << write_size;
    }
}

bool DaemonSerialPortWrapper::open(QIODevice::OpenMode mode)
{
    return m_serialPort.open(mode);
}

void DaemonSerialPortWrapper::close()
{
    m_serialPort.close();
}
