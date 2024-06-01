#ifndef CONTROL_PANEL_PERCEPTIONS_TAB_H
#define CONTROL_PANEL_PERCEPTIONS_TAB_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <video_analyzer/msg/video_analysis.hpp>
#include <audio_analyzer/msg/audio_analysis.hpp>
#include <std_msgs/msg/empty.hpp>
#include <person_identification/msg/person_names.hpp>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class PerceptionsTab : public QWidget
{
    Q_OBJECT

    rclcpp::Node::SharedPtr m_node;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_analyzedImage3dSubscriber;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_analyzedImage2dWideSubscriber;
    rclcpp::Subscription<video_analyzer::msg::VideoAnalysis>::SharedPtr m_videoAnalysis2dWideSubscriber;
    rclcpp::Subscription<audio_analyzer::msg::AudioAnalysis>::SharedPtr m_audioAnalysisSubscriber;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_robotNameDetectedSubscriber;
    rclcpp::Subscription<person_identification::msg::PersonNames>::SharedPtr m_personNamesSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_videoAnalyzer3dDesireId;
    QVariant m_videoAnalyzer2dWideDesireId;
    QVariant m_audioAnalyzerDesireId;
    QVariant m_robotNameDetectorDesireId;

    bool m_camera2dWideEnabled;

public:
    PerceptionsTab(
        rclcpp::Node::SharedPtr node,
        std::shared_ptr<DesireSet> desireSet,
        bool camera2dWideEnabled,
        QWidget* parent = nullptr);

private slots:
    void onVideoAnalyzer3dButtonToggled(bool checked);
    void onVideoAnalyzer2dWideButtonToggled(bool checked);
    void onAudioAnalyzerButtonToggled(bool checked);
    void onRobotNameDetectorButtonToggled(bool checked);

private:
    void analyzedImage3dSubscriberCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void analyzedImage2dWideSubscriberCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void videoAnalysis2dWideSubscriberCallback(const video_analyzer::msg::VideoAnalysis::SharedPtr msg);
    void audioAnalysisSubscriberCallback(const audio_analyzer::msg::AudioAnalysis::SharedPtr msg);
    void robotNameDetectedSubscriberCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void personNamesSubscriberCallback(const person_identification::msg::PersonNames::SharedPtr msg);

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
