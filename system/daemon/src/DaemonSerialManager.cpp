#include "DaemonSerialManager.h"
#include <QDebug>

DaemonSerialManager::DaemonSerialManager(const QSerialPortInfo &port, QObject *parent)
    : QObject(parent), m_serialPort(nullptr)
{

    m_serialPort = new QSerialPort(port, this);

    // TODO port setup, hardcoded for now
    m_serialPort->setDataBits(QSerialPort::Data8);
    m_serialPort->setStopBits(QSerialPort::OneStop);
    m_serialPort->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
    m_serialPort->setFlowControl(QSerialPort::NoFlowControl);

    if (m_serialPort->open(QIODevice::ReadWrite))
    {
        // Connect signals
        connect(m_serialPort, &QSerialPort::errorOccurred, this, &DaemonSerialManager::onErrorOccurred);
        connect(m_serialPort, &QSerialPort::readyRead, this, &DaemonSerialManager::onReadyRead);
    }
    else
    {
        qDebug() << "Cannot open port: " << port.portName();
    }
}

QList<QSerialPortInfo> DaemonSerialManager::availablePorts()
{
    return QSerialPortInfo::availablePorts();
}

bool DaemonSerialManager::isValidPort(const QString &name)
{
    for (auto &&port : DaemonSerialManager::availablePorts())
    {
        if(port.portName() == name) {
            return true;
        }
    }

    return false;
}

void DaemonSerialManager::printAvailablePorts()
{
    for (auto &&port : DaemonSerialManager::availablePorts())
    {
        qDebug() << port.portName() << " " << port.manufacturer();

    }
}

void DaemonSerialManager::onErrorOccurred(QSerialPort::SerialPortError error)
{
    qDebug() << "DaemonSerialManager::onErrorOccurred" << error;
}

void DaemonSerialManager::onReadyRead()
{
    QByteArray data = m_serialPort->readAll();
    qDebug() << "readyRead: " << data;
}

