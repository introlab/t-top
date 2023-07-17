#ifndef _DAEMON_SERIAL_PORT_WRAPPER_H_
#define _DAEMON_SERIAL_PORT_WRAPPER_H_

#include <QObject>
#include <SerialCommunication.h>
#include <QSerialPort>
#include <algorithm>

class DaemonSerialPortWrapper : public QObject, public SerialPort
{
    Q_OBJECT

public:
    DaemonSerialPortWrapper(const QSerialPortInfo& info, QObject* parent = nullptr);
    void read(SerialCommunicationBufferView& buffer) override;
    void write(const uint8_t* data, size_t size) override;
    bool open(QSerialPort::OpenMode mode);
    void close();

signals:

    void errorOccurred(QSerialPort::SerialPortError error);
    void readyRead();

private:
    QSerialPort m_serialPort;
};

#endif  // _DAEMON_SERIAL_PORT_WRAPPER_H_
