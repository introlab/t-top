#ifndef CONTROL_PANEL_PERCEPTIONS_TAB_H
#define CONTROL_PANEL_PERCEPTIONS_TAB_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <video_analyzer/VideoAnalysis.h>
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
    ros::Subscriber m_analyzedImage3dSubscriber;
    ros::Subscriber m_analyzedImage2dWideSubscriber;
    ros::Subscriber m_videoAnalysis2dWideSubscriber;
    ros::Subscriber m_audioAnalysisSubscriber;
    ros::Subscriber m_robotNameDetectedSubscriber;
    ros::Subscriber m_personNamesSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_videoAnalyzer3dDesireId;
    QVariant m_videoAnalyzer2dWideDesireId;
    QVariant m_audioAnalyzerDesireId;
    QVariant m_robotNameDetectorDesireId;

    bool m_camera2dWideEnabled;

public:
    PerceptionsTab(
        ros::NodeHandle& nodeHandle,
        std::shared_ptr<DesireSet> desireSet,
        bool camera2dWideEnabled,
        QWidget* parent = nullptr);

private slots:
    void onVideoAnalyzer3dButtonToggled(bool checked);
    void onVideoAnalyzer2dWideButtonToggled(bool checked);
    void onAudioAnalyzerButtonToggled(bool checked);
    void onRobotNameDetectorButtonToggled(bool checked);

private:
    void analyzedImage3dSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg);
    void analyzedImage2dWideSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg);
    void videoAnalysis2dWideSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg);
    void audioAnalysisSubscriberCallback(const audio_analyzer::AudioAnalysis::ConstPtr& msg);
    void robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg);
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);

    void createUi();

    template<class D>
    void toggleDesire(bool checked, QVariant& desireId);

    // UI members
    QPushButton* m_videoAnalyzer3dButton;
    QPushButton* m_videoAnalyzer2dWideButton;
    QPushButton* m_audioAnalyzerButton;
    QPushButton* m_robotNameDetectorButton;

    ImageDisplay* m_videoAnalyzer3dImageDisplay;
    ImageDisplay* m_videoAnalyzer2dWideImageDisplay;
    ImageDisplay* m_videoAnalyzer2dWideSegmentationcImageDisplay;
    QLineEdit* m_soundClassesLineEdit;
    QLineEdit* m_robotNameDetectionTimeLineEdit;
    QLineEdit* m_identifiedPersonsLineEdit;
};

template<class D>
void PerceptionsTab::toggleDesire(bool checked, QVariant& desireId)
{
    if (checked)
    {
        auto desire = std::make_unique<D>();
        desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (desireId.isValid())
    {
        m_desireSet->removeDesire(desireId.toULongLong());
        desireId.clear();
    }
}

#endif
