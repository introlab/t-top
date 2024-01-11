#ifndef CONNECT4_CONNECT4_WIDGET_H
#define CONNECT4_CONNECT4_WIDGET_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QWebSocket>
#include <QNetworkReply>
#include <QTimer>

#include <ros/ros.h>
#include <hbba_lite/core/DesireSet.h>
#include <std_msgs/Empty.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>

#include <atomic>
#include <memory>

class Connect4Widget : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    std::shared_ptr<DesireSet> m_desireSet;

    std::atomic_bool m_enabled;

    ros::Subscriber m_startButtonPressedSub;
    ros::Subscriber m_stopButtonPressedSub;
    ros::Subscriber m_remoteImageSub;

    ros::Publisher  m_volumePub;
    QTimer* m_setVolumeTimer;

    ros::Subscriber m_openteraEventSubscriber;
    QString m_connect4ManagerWebSocketUrl;
    QString m_connect4ManagerWebSocketPassword;
    QString m_observedParticipantName;
    bool m_connect4ManagerConnectionRequested;

    QTimer* m_connect4ManagerWebSocketTimer;
    QWebSocket* m_connect4ManagerWebSocket;

public:
    Connect4Widget(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private Q_SLOTS:
    void onSetVolumeTimerTimeout();

    void onConnect4ManagerWebSocketTimeout();

    void onConnect4ManagerWebSocketSslErrors(const QList<QSslError>& errors);
    void onConnect4ManagerWebSocketConnected();
    void onConnect4ManagerWebSocketDisconnected();
    void onConnect4ManagerWebSocketErrorOccurred(QAbstractSocket::SocketError error);
    void onConnect4ManagerWebSocketTextMessageReceived(const QString& message);

private:
    void startButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void stopButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void remoteImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);

    void setVolume(float volume);

    void openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg);
    bool sendConnect4ManagerEvent(const QString& event, const QJsonObject& data);

    QString getParticipantName(const std::string& deviceName, const std::string& sessionParameters);
    void parseSessionUrl(const std::string& sessionUrl, QString& webSocketUrl, QString& password);

    void handleGameFinishedEvent(const QString& result);
    void addRotatingSinDesire(uint8_t r, uint8_t g, uint8_t b);

private:
    ImageDisplay* m_imageDisplay;
};
#endif
