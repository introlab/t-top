#ifndef _DAEMON_SERIAL_MANAGER_H_
#define _DAEMON_SERIAL_MANAGER_H_

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>

// From common library
#include "SerialCommunication.h"
#include "SerialCommunicationBuffer.h"
#include "SerialMessagePayloads.h"
#include "SerialMessages.h"

class DaemonSerialManager : public QObject {

    Q_OBJECT

public:

    DaemonSerialManager(const QSerialPortInfo &port, QObject *parent=nullptr);

    static QList<QSerialPortInfo> availablePorts();
    static bool isValidPort(const QString &name);
    static void printAvailablePorts();


private slots:
     void onErrorOccurred(QSerialPort::SerialPortError error);
     void onReadyRead();

private:
    QSerialPort *m_serialPort;

};


#endif // _DAEMON_SERIAL_MANAGER_H_
