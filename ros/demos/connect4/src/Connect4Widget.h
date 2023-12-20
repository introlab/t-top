#ifndef CONNECT4_CONNECT4_WIDGET_H
#define CONNECT4_CONNECT4_WIDGET_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QTimer>

#include <ros/ros.h>
#include <hbba_lite/core/DesireSet.h>
#include <std_msgs/Empty.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>

#include <OpenteraWebrtcNativeClient/DataChannelClient.h>

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
    std::string m_deviceName;
    std::string m_participantName;
    std::unique_ptr<opentera::DataChannelClient> m_gameDataChannelClient;

public:
    Connect4Widget(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private Q_SLOTS:
    void onSetVolumeTimerTimeout();

private:
    void startButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void stopButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void remoteImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);

    void setVolume(float volume);

    void openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg);

    void connectGameDataChannel(
        const std::string& deviceName,
        const std::string& sessionUrl,
        const std::string& sessionParameters);
    std::string getParticipantName(const std::string& deviceName, const std::string& sessionParameters);
    void parseSessionUrl(const std::string& sessionUrl, std::string& baseUrl, std::string& password);
    void setGameDataChannelCallbacks();

    void handleGameMessage(const QString& message);
    bool isWinner(const std::string& participantId);
    void addRotatingSinDesire(uint8_t r, uint8_t g, uint8_t b);

    void closeGameDataChannel();

private:
    ImageDisplay* m_imageDisplay;
};
#endif
