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

vector<QColor> getSemanticSegmentationPaletteFromClassCount(size_t classCount)
{
    constexpr int MAX_PIXEL_VALUE = 255;
    vector<QColor> palette(classCount);
    int colorStep = static_cast<int>(ceil(MAX_PIXEL_VALUE / pow(classCount, 1.f / 3.f)));
    int r = 0;
    int g = 0;
    int b = 0;
    for (size_t i = 0; i < classCount; i++)
    {
        palette[i] = QColor(r, g, b);

        r += colorStep;
        if (r > MAX_PIXEL_VALUE)
        {
            r = 0;
            g += colorStep;
            if (g > MAX_PIXEL_VALUE)
            {
                g = 0;
                b += colorStep;
            }
        }
    }

    return palette;
}

QImage semanticSegmentationToImage(
    rclcpp::Node& node,
    const video_analyzer::msg::SemanticSegmentation& semanticSegmentation)
{
    int width = semanticSegmentation.width;
    int height = semanticSegmentation.height;
    if (semanticSegmentation.class_indexes.size() != width * height)
    {
        RCLCPP_ERROR(node.get_logger(), "Invalid semantic segmentation (class_indexes.size() != width * height)");
        return QImage();
    }

    QImage image(width, height, QImage::Format_RGB32);
    image.fill(QColor(0, 0, 0));
    vector<QColor> palette = getSemanticSegmentationPaletteFromClassCount(semanticSegmentation.class_names.size());

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int classIndex = semanticSegmentation.class_indexes[y * width + x];
            if (classIndex < palette.size())
            {
                image.setPixelColor(x, y, palette[classIndex]);
            }
        }
    }

    return image;
}

PerceptionsTab::PerceptionsTab(
    rclcpp::Node::SharedPtr node,
    shared_ptr<DesireSet> desireSet,
    bool camera2dWideEnabled,
    QWidget* parent)
    : QWidget(parent),
      m_node(std::move(node)),
      m_desireSet(std::move(desireSet)),
      m_camera2dWideEnabled(camera2dWideEnabled)
{
    createUi();

    m_analyzedImage3dSubscriber = m_node->create_subscription<sensor_msgs::msg::Image>(
        "camera_3d/analysed_image",
        1,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) { analyzedImage3dSubscriberCallback(msg); });
    if (m_camera2dWideEnabled)
    {
        m_analyzedImage2dWideSubscriber = m_node->create_subscription<sensor_msgs::msg::Image>(
            "camera_2d_wide/analysed_image",
            1,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { analyzedImage2dWideSubscriberCallback(msg); });
        m_videoAnalysis2dWideSubscriber = m_node->create_subscription<video_analyzer::msg::VideoAnalysis>(
            "camera_2d_wide/video_analysis",
            1,
            [this](const video_analyzer::msg::VideoAnalysis::SharedPtr msg)
            { videoAnalysis2dWideSubscriberCallback(msg); });
    }

    m_audioAnalysisSubscriber = m_node->create_subscription<audio_analyzer::msg::AudioAnalysis>(
        "audio_analysis",
        1,
        [this](const audio_analyzer::msg::AudioAnalysis::SharedPtr msg) { audioAnalysisSubscriberCallback(msg); });
    m_robotNameDetectedSubscriber = m_node->create_subscription<std_msgs::msg::Empty>(
        "robot_name_detected",
        1,
        [this](const std_msgs::msg::Empty::SharedPtr msg) { robotNameDetectedSubscriberCallback(msg); });
    m_personNamesSubscriber = m_node->create_subscription<person_identification::msg::PersonNames>(
        "person_names",
        1,
        [this](const person_identification::msg::PersonNames::SharedPtr msg) { personNamesSubscriberCallback(msg); });
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

void PerceptionsTab::analyzedImage3dSubscriberCallback(const sensor_msgs::msg::Image::SharedPtr msg)
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

void PerceptionsTab::analyzedImage2dWideSubscriberCallback(const sensor_msgs::msg::Image::SharedPtr msg)
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

void PerceptionsTab::videoAnalysis2dWideSubscriberCallback(const video_analyzer::msg::VideoAnalysis::SharedPtr msg)
{
    if (msg->semantic_segmentation.empty())
    {
        return;
    }

    invokeLater(
        [this, msg]()
        {
            m_videoAnalyzer2dWideSegmentationcImageDisplay->setImage(
                semanticSegmentationToImage(*m_node, msg->semantic_segmentation[0]));
        });
}

void PerceptionsTab::audioAnalysisSubscriberCallback(const audio_analyzer::msg::AudioAnalysis::SharedPtr msg)
{
    QString classes = mergeStdStrings(msg->audio_classes);
    invokeLater([this, classes]() { m_soundClassesLineEdit->setText(classes); });
}

void PerceptionsTab::robotNameDetectedSubscriberCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    auto currentTime = QDateTime::currentDateTime();
    invokeLater([this, currentTime]() { m_robotNameDetectionTimeLineEdit->setText(currentTime.toString()); });
}

void PerceptionsTab::personNamesSubscriberCallback(const person_identification::msg::PersonNames::SharedPtr msg)
{
    vector<string> names;
    names.reserve(msg->names.size());

    transform(
        msg->names.begin(),
        msg->names.end(),
        back_inserter(names),
        [](const person_identification::msg::PersonName& name) { return name.name; });

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
        m_videoAnalyzer2dWideSegmentationcImageDisplay = new ImageDisplay;
    }
    else
    {
        m_videoAnalyzer2dWideButton = nullptr;
        m_videoAnalyzer2dWideImageDisplay = nullptr;
        m_videoAnalyzer2dWideSegmentationcImageDisplay = nullptr;
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
    globalLayout->addWidget(new QLabel("Video Analyzer 3D Image:"));
    globalLayout->addWidget(m_videoAnalyzer3dImageDisplay, 1);
    if (m_videoAnalyzer2dWideImageDisplay != nullptr)
    {
        globalLayout->addWidget(new QLabel("Video Analyzer 2D Image:"));
        globalLayout->addWidget(m_videoAnalyzer2dWideImageDisplay, 1);
        globalLayout->addWidget(m_videoAnalyzer2dWideSegmentationcImageDisplay, 1);
    }
    globalLayout->addWidget(new QLabel("Sound Classes:"));
    globalLayout->addWidget(m_soundClassesLineEdit);
    globalLayout->addWidget(new QLabel("Last Robot Name Detection Time:"));
    globalLayout->addWidget(m_robotNameDetectionTimeLineEdit);
    globalLayout->addWidget(new QLabel("Identified Persons:"));
    globalLayout->addWidget(m_identifiedPersonsLineEdit);

    setLayout(globalLayout);
}
