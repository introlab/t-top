#ifndef CONNECT4_CONNECT4_WIDGET_H
#define CONNECT4_CONNECT4_WIDGET_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QWebSocket>
#include <QNetworkReply>
#include <QTimer>

#include <rclcpp/rclcpp.hpp>
#include <hbba_lite/core/DesireSet.h>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_image.hpp>
#include <opentera_webrtc_ros_msgs/msg/open_tera_event.hpp>

#include <atomic>
#include <memory>

class Connect4Widget : public QWidget
{
    Q_OBJECT

    rclcpp::Node::SharedPtr m_node;
    std::shared_ptr<DesireSet> m_desireSet;

    std::atomic_bool m_enabled;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_startButtonPressedSub;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_stopButtonPressedSub;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerImage>::SharedPtr m_remoteImageSub;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>::SharedPtr m_openteraEventSubscriber;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr  m_volumePub;

    QTimer* m_setVolumeTimer;


    QString m_connect4ManagerWebSocketUrl;
    QString m_connect4ManagerWebSocketPassword;
    QString m_observedParticipantName;
    bool m_connect4ManagerConnectionRequested;

    QTimer* m_connect4ManagerWebSocketTimer;
    QWebSocket* m_connect4ManagerWebSocket;

public:
    Connect4Widget(rclcpp::Node::SharedPtr node, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private Q_SLOTS:
    void onSetVolumeTimerTimeout();

    void onConnect4ManagerWebSocketTimeout();

    void onConnect4ManagerWebSocketSslErrors(const QList<QSslError>& errors);
    void onConnect4ManagerWebSocketConnected();
    void onConnect4ManagerWebSocketDisconnected();
    void onConnect4ManagerWebSocketErrorOccurred(QAbstractSocket::SocketError error);
    void onConnect4ManagerWebSocketTextMessageReceived(const QString& message);

private:
    void startButtonPressedCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void stopButtonPressedCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void remoteImageCallback(const opentera_webrtc_ros_msgs::msg::PeerImage::SharedPtr msg);

    void setVolume(float volume);

    void openteraEventCallback(const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::SharedPtr msg);
    bool sendConnect4ManagerEvent(const QString& event, const QJsonObject& data);

    QString getParticipantName(const std::string& deviceName, const std::string& sessionParameters);
    void parseSessionUrl(const std::string& sessionUrl, QString& webSocketUrl, QString& password);

    void handleGameFinishedEvent(const QString& result);
    void addRotatingSinDesire(uint8_t r, uint8_t g, uint8_t b);

private:
    ImageDisplay* m_imageDisplay;
};
#endif
