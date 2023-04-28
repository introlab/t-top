#ifndef CONTROL_PANEL_CONTROL_PANEL_H
#define CONTROL_PANEL_CONTROL_PANEL_H

#include "AvatarTab.h"
#include "SpeechTab.h"
#include "GestureTab.h"
#include "BehaviorsTab.h"
#include "LedTab.h"
#include "PerceptionsTab.h"

#include <QWidget>
#include <QTabWidget>
#include <QLCDNumber>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <daemon_ros_client/BaseStatus.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class ControlPanel : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_volumePublisher;
    ros::Subscriber m_baseStatusSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;

public:
    ControlPanel(
        ros::NodeHandle& nodeHandle,
        std::shared_ptr<DesireSet> desireSet,
        bool camera2dWideEnabled,
        QWidget* parent = nullptr);

private slots:
    void onVolumeChanged(int volume);

private:
    void baseStatusSubscriberCallback(const daemon_ros_client::BaseStatus::ConstPtr& msg);

    void createUi(bool camera2dWideEnabled);

    // UI members
    AvatarTab* m_avatarTab;
    SpeechTab* m_speechTab;
    GestureTab* m_gestureTab;
    BehaviorsTab* m_behaviorsTab;
    LedTab* m_ledTab;
    PerceptionsTab* m_perceptionsTab;

    QTabWidget* m_tabWidget;

    QSlider* m_volumeSlider;
    QLCDNumber* m_batteryLevel;
};

#endif
