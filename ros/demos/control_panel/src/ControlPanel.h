#ifndef CONTROL_PANEL_CONTROL_PANEL_H
#define CONTROL_PANEL_CONTROL_PANEL_H

#include "AvatarTab.h"
#include "SpeechTab.h"
#include "GestureTab.h"
#include "BehaviorsTab.h"
#include "PerceptionsTab.h"

#include <QWidget>
#include <QTabWidget>
#include <QLCDNumber>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

class ControlPanel : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_volumePublisher;
    ros::Subscriber m_batterySubscriber;

public:
    ControlPanel(ros::NodeHandle& nodeHandle, QWidget* parent = nullptr);

private slots:
    void onVolumeChanged(int volume);

private:
    void batterySubscriberCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);

    void createUi();

    // UI members
    AvatarTab* m_avatarTab;
    SpeechTab* m_speechTab;
    GestureTab* m_gestureTab;
    BehaviorsTab* m_behaviorsTab;
    PerceptionsTab* m_perceptionsTab;

    QTabWidget* m_tabWidget;

    QSlider* m_volumeSlider;
    QLCDNumber* m_batteryLevel;
};

#endif
