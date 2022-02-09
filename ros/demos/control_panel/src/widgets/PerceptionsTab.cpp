#include "PerceptionsTab.h"
#include "../QtUtils.h"

#include <QDebug>
#include <QDateTime>
#include <QLabel>
#include <QVBoxLayout>

#include <t_top/hbba_lite/Desires.h>

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

PerceptionsTab::PerceptionsTab(ros::NodeHandle& nodeHandle, shared_ptr<DesireSet> desireSet, QWidget* parent) :
        QWidget(parent), m_nodeHandle(nodeHandle), m_desireSet(std::move(desireSet))
{
    createUi();

    m_analyzedImageSubscriber = nodeHandle.subscribe("analysed_image", 1,
        &PerceptionsTab::analyzedImageSubscriberCallback, this);
    m_audioAnalysisSubscriber = nodeHandle.subscribe("audio_analysis", 1,
        &PerceptionsTab::audioAnalysisSubscriberCallback, this);
    m_robotNameDetectedSubscriber = nodeHandle.subscribe("robot_name_detected", 1,
        &PerceptionsTab::robotNameDetectedSubscriberCallback, this);
    m_personNamesSubscriber = nodeHandle.subscribe("person_names", 1,
        &PerceptionsTab::personNamesSubscriberCallback, this);
}

void PerceptionsTab::onVideoAnalyzerButtonToggled(bool checked)
{
    if (checked)
    {
        auto desire = make_unique<FastVideoAnalyzerWithAnalyzedImageDesire>();
        m_videoAnalyzerDesireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_videoAnalyzerDesireId.isValid())
    {
        m_desireSet->removeDesire(m_videoAnalyzerDesireId.toULongLong());
        m_videoAnalyzerDesireId.clear();
    }
}

void PerceptionsTab::onAudioAnalyzerButtonToggled(bool checked)
{
    if (checked)
    {
        auto desire = make_unique<AudioAnalyzerDesire>();
        m_audioAnalyzerDesireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_audioAnalyzerDesireId.isValid())
    {
        m_desireSet->removeDesire(m_audioAnalyzerDesireId.toULongLong());
        m_audioAnalyzerDesireId.clear();
    }
}

void PerceptionsTab::onRobotNameDetectorButtonToggled(bool checked)
{
    if (checked)
    {
        auto desire = make_unique<RobotNameDetectorDesire>();
        m_robotNameDetectorDesireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_robotNameDetectorDesireId.isValid())
    {
        m_desireSet->removeDesire(m_robotNameDetectorDesireId.toULongLong());
        m_robotNameDetectorDesireId.clear();
    }
}

void PerceptionsTab::analyzedImageSubscriberCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    if (msg->encoding != "rgb8")
    {
        return;
    }

    invokeLater([this, msg]()
    {
        m_videoAnalyzerImageDisplay->setImage(QImage(msg->data.data(), msg->width, msg->height, QImage::Format_RGB888));
    });
}

void PerceptionsTab::audioAnalysisSubscriberCallback(const audio_analyzer::AudioAnalysis::ConstPtr& msg)
{
    QString classes = mergeStdStrings(msg->audio_classes);
    invokeLater([this, classes]()
    {
        m_soundClassesLineEdit->setText(classes);
    });
}

void PerceptionsTab::robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg)
{
    auto currentTime = QDateTime::currentDateTime();
    invokeLater([this, currentTime]()
    {
        m_robotNameDetectionTimeLineEdit->setText(currentTime.toString());
    });
}

void PerceptionsTab::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
{
    vector<string> names(msg->names.size());
    transform(msg->names.begin(), msg->names.end(), back_inserter(names),
        [](const person_identification::PersonName& name) { return name.name; });

    QString mergedNames = mergeStdStrings(names);
    invokeLater([this, mergedNames]()
    {
        m_identifiedPersonsLineEdit->setText(mergedNames);
    });
}

void PerceptionsTab::createUi()
{
    m_videoAnalyzerButton = new QPushButton("Video Analyzer");
    m_videoAnalyzerButton->setCheckable(true);
    connect(m_videoAnalyzerButton, &QPushButton::toggled, this, &PerceptionsTab::onVideoAnalyzerButtonToggled);

    m_audioAnalyzerButton = new QPushButton("Audio Analyzer");
    m_audioAnalyzerButton->setCheckable(true);
    connect(m_audioAnalyzerButton, &QPushButton::toggled, this, &PerceptionsTab::onAudioAnalyzerButtonToggled);

    m_robotNameDetectorButton = new QPushButton("Robot Name Detector");
    m_robotNameDetectorButton->setCheckable(true);
    connect(m_robotNameDetectorButton, &QPushButton::toggled, this, &PerceptionsTab::onRobotNameDetectorButtonToggled);

    m_videoAnalyzerImageDisplay = new ImageDisplay;

    m_soundClassesLineEdit = new QLineEdit;
    m_soundClassesLineEdit->setReadOnly(true);

    m_robotNameDetectionTimeLineEdit = new QLineEdit;
    m_robotNameDetectionTimeLineEdit->setReadOnly(true);

    m_identifiedPersonsLineEdit = new QLineEdit;
    m_identifiedPersonsLineEdit->setReadOnly(true);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_videoAnalyzerButton);
    globalLayout->addWidget(m_audioAnalyzerButton);
    globalLayout->addWidget(m_robotNameDetectorButton);
    globalLayout->addWidget(new QLabel("Video Analyzer Image:"));
    globalLayout->addWidget(m_videoAnalyzerImageDisplay, 1);
    globalLayout->addWidget(new QLabel("Sound Classes:"));
    globalLayout->addWidget(m_soundClassesLineEdit);
    globalLayout->addWidget(new QLabel("Last Robot Name Detection Time:"));
    globalLayout->addWidget(m_robotNameDetectionTimeLineEdit);
    globalLayout->addWidget(new QLabel("Identified Person:"));
    globalLayout->addWidget(m_identifiedPersonsLineEdit);

    setLayout(globalLayout);
}
