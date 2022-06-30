#include "PerceptionsTab.h"
#include "../QtUtils.h"

#include <QDebug>
#include <QDateTime>
#include <QLabel>
#include <QVBoxLayout>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

QString mergeStdStrings(const vector<string> values)
{
    QString mergedValues;
    for (size_t i = 0; i < values.size(); i++)
    {
        mergedValues.append(QString::fromStdString(values[i]));
        if (i < values.size() - 1)
        {
            mergedValues.append(", ");
        }
    }

    return mergedValues;
}

PerceptionsTab::PerceptionsTab(
    ros::NodeHandle& nodeHandle,
    shared_ptr<DesireSet> desireSet,
    bool camera2dWideEnabled,
    QWidget* parent)
    : QWidget(parent),
      m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet)),
      m_camera2dWideEnabled(camera2dWideEnabled)
{
    createUi();

    m_analyzedImage3dSubscriber =
        nodeHandle.subscribe("camera_3d/analysed_image", 1, &PerceptionsTab::analyzedImage3dSubscriberCallback, this);
    if (m_camera2dWideEnabled)
    {
        m_analyzedImage2dWideSubscriber = nodeHandle.subscribe(
            "camera_2d_wide/analysed_image",
            1,
            &PerceptionsTab::analyzedImage2dWideSubscriberCallback,
            this);
    }

    m_audioAnalysisSubscriber =
        nodeHandle.subscribe("audio_analysis", 1, &PerceptionsTab::audioAnalysisSubscriberCallback, this);
    m_robotNameDetectedSubscriber =
        nodeHandle.subscribe("robot_name_detected", 1, &PerceptionsTab::robotNameDetectedSubscriberCallback, this);
    m_personNamesSubscriber =
        nodeHandle.subscribe("person_names", 1, &PerceptionsTab::personNamesSubscriberCallback, this);
}

void PerceptionsTab::onVideoAnalyzer3dButtonToggled(bool checked)
{
    toggleDesire<FastVideoAnalyzer3dWithAnalyzedImageDesire>(checked, m_videoAnalyzer3dDesireId);
}

void PerceptionsTab::onVideoAnalyzer2dWideButtonToggled(bool checked)
{
    toggleDesire<FastVideoAnalyzer2dWideWithAnalyzedImageDesire>(checked, m_videoAnalyzer2dWideDesireId);
}

void PerceptionsTab::onAudioAnalyzerButtonToggled(bool checked)
{
    toggleDesire<AudioAnalyzerDesire>(checked, m_audioAnalyzerDesireId);
}

void PerceptionsTab::onRobotNameDetectorButtonToggled(bool checked)
{
    toggleDesire<RobotNameDetectorDesire>(checked, m_robotNameDetectorDesireId);
}

void PerceptionsTab::analyzedImage3dSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    if (msg->encoding != "rgb8")
    {
        return;
    }

    invokeLater(
        [this, msg]() {
            m_videoAnalyzer3dImageDisplay->setImage(
                QImage(msg->data.data(), msg->width, msg->height, QImage::Format_RGB888));
        });
}

void PerceptionsTab::analyzedImage2dWideSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    if (msg->encoding != "rgb8")
    {
        return;
    }

    invokeLater(
        [this, msg]()
        {
            m_videoAnalyzer2dWideImageDisplay->setImage(
                QImage(msg->data.data(), msg->width, msg->height, QImage::Format_RGB888));
        });
}

void PerceptionsTab::audioAnalysisSubscriberCallback(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
{
    QString classes = mergeStdStrings(msg->audio_classes);
    invokeLater([this, classes]() { m_soundClassesLineEdit->setText(classes); });
}

void PerceptionsTab::robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg)
{
    auto currentTime = QDateTime::currentDateTime();
    invokeLater([this, currentTime]() { m_robotNameDetectionTimeLineEdit->setText(currentTime.toString()); });
}

void PerceptionsTab::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
{
    vector<string> names;
    names.reserve(msg->names.size());

    transform(
        msg->names.begin(),
        msg->names.end(),
        back_inserter(names),
        [](const person_identification::PersonName& name) { return name.name; });

    QString mergedNames = mergeStdStrings(names);
    invokeLater([this, mergedNames]() { m_identifiedPersonsLineEdit->setText(mergedNames); });
}

void PerceptionsTab::createUi()
{
    m_videoAnalyzer3dButton = new QPushButton("Video Analyzer 3D");
    m_videoAnalyzer3dButton->setCheckable(true);
    connect(m_videoAnalyzer3dButton, &QPushButton::toggled, this, &PerceptionsTab::onVideoAnalyzer3dButtonToggled);

    m_audioAnalyzerButton = new QPushButton("Audio Analyzer");
    m_audioAnalyzerButton->setCheckable(true);
    connect(m_audioAnalyzerButton, &QPushButton::toggled, this, &PerceptionsTab::onAudioAnalyzerButtonToggled);

    m_robotNameDetectorButton = new QPushButton("Robot Name Detector");
    m_robotNameDetectorButton->setCheckable(true);
    connect(m_robotNameDetectorButton, &QPushButton::toggled, this, &PerceptionsTab::onRobotNameDetectorButtonToggled);

    m_videoAnalyzer3dImageDisplay = new ImageDisplay;

    if (m_camera2dWideEnabled)
    {
        m_videoAnalyzer2dWideButton = new QPushButton("Video Analyzer 2D Wide");
        m_videoAnalyzer2dWideButton->setCheckable(true);
        connect(
            m_videoAnalyzer2dWideButton,
            &QPushButton::toggled,
            this,
            &PerceptionsTab::onVideoAnalyzer2dWideButtonToggled);

        m_videoAnalyzer2dWideImageDisplay = new ImageDisplay;
    }
    else
    {
        m_videoAnalyzer2dWideButton = nullptr;
        m_videoAnalyzer2dWideImageDisplay = nullptr;
    }

    m_soundClassesLineEdit = new QLineEdit;
    m_soundClassesLineEdit->setReadOnly(true);

    m_robotNameDetectionTimeLineEdit = new QLineEdit;
    m_robotNameDetectionTimeLineEdit->setReadOnly(true);

    m_identifiedPersonsLineEdit = new QLineEdit;
    m_identifiedPersonsLineEdit->setReadOnly(true);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_videoAnalyzer3dButton);
    if (m_videoAnalyzer2dWideButton != nullptr)
    {
        globalLayout->addWidget(m_videoAnalyzer2dWideButton);
    }
    globalLayout->addWidget(m_audioAnalyzerButton);
    globalLayout->addWidget(m_robotNameDetectorButton);
    globalLayout->addWidget(new QLabel("Video Analyzer Image:"));
    globalLayout->addWidget(m_videoAnalyzer3dImageDisplay, 1);
    if (m_videoAnalyzer2dWideImageDisplay != nullptr)
    {
        globalLayout->addWidget(m_videoAnalyzer2dWideImageDisplay, 1);
    }
    globalLayout->addWidget(new QLabel("Sound Classes:"));
    globalLayout->addWidget(m_soundClassesLineEdit);
    globalLayout->addWidget(new QLabel("Last Robot Name Detection Time:"));
    globalLayout->addWidget(m_robotNameDetectionTimeLineEdit);
    globalLayout->addWidget(new QLabel("Identified Person:"));
    globalLayout->addWidget(m_identifiedPersonsLineEdit);

    setLayout(globalLayout);
}
