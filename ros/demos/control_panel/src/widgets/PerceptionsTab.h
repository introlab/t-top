#ifndef CONTROL_PANEL_PERCEPTIONS_TAB_H
#define CONTROL_PANEL_PERCEPTIONS_TAB_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <audio_analyzer/AudioAnalysis.h>
#include <std_msgs/Empty.h>
#include <person_identification/PersonNames.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class PerceptionsTab : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    ros::Subscriber m_analyzedImageSubscriber;
    ros::Subscriber m_audioAnalysisSubscriber;
    ros::Subscriber m_robotNameDetectedSubscriber;
    ros::Subscriber m_personNamesSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_videoAnalyzerDesireId;
    QVariant m_audioAnalyzerDesireId;
    QVariant m_robotNameDetectorDesireId;

public:
    PerceptionsTab(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private slots:
    void onVideoAnalyzerButtonToggled(bool checked);
    void onAudioAnalyzerButtonToggled(bool checked);
    void onRobotNameDetectorButtonToggled(bool checked);

private:
    void analyzedImageSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg);
    void audioAnalysisSubscriberCallback(const audio_analyzer::AudioAnalysis::ConstPtr& msg);
    void robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg);
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);

    void createUi();

    // UI members
    QPushButton* m_videoAnalyzerButton;
    QPushButton* m_audioAnalyzerButton;
    QPushButton* m_robotNameDetectorButton;

    ImageDisplay* m_videoAnalyzerImageDisplay;
    QLineEdit* m_soundClassesLineEdit;
    QLineEdit* m_robotNameDetectionTimeLineEdit;
    QLineEdit* m_identifiedPersonsLineEdit;
};

#endif
