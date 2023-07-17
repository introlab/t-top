#include "ControlPanel.h"
#include "../QtUtils.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

#include <std_msgs/UInt8.h>

using namespace std;

constexpr int AUDIO_POWER_AMPLIFIER_MIN_VOLUME = 0;
constexpr int AUDIO_POWER_AMPLIFIER_MAX_VOLUME = 63;
constexpr int AUDIO_POWER_AMPLIFIER_DEFAULT_VOLUME = 24;

ControlPanel::ControlPanel(
    ros::NodeHandle& nodeHandle,
    shared_ptr<DesireSet> desireSet,
    bool camera2dWideEnabled,
    QWidget* parent)
    : QWidget(parent),
      m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet))
{
    m_volumePublisher = nodeHandle.advertise<std_msgs::UInt8>("daemon/set_volume", 1);

    createUi(camera2dWideEnabled);

    m_baseStatusSubscriber =
        nodeHandle.subscribe("daemon/base_status", 1, &ControlPanel::baseStatusSubscriberCallback, this);
}

void ControlPanel::onVolumeChanged(int volume)
{
    std_msgs::UInt8 msg;
    msg.data = volume;
    m_volumePublisher.publish(msg);
}

void ControlPanel::baseStatusSubscriberCallback(const daemon_ros_client::BaseStatus::ConstPtr& msg)
{
    int stateOfCharge = static_cast<int>(msg->state_of_charge);
    int volume = msg->volume;
    int maximumVolume = msg->maximum_volume;

    invokeLater(
        [=]()
        {
            m_batteryLevel->display(QString::number(stateOfCharge));

            QSignalBlocker volumeSliderSignalBlocker(m_volumeSlider);
            m_volumeSlider->setMaximum(maximumVolume);
            m_volumeSlider->setValue(volume);
        });
}

void ControlPanel::createUi(bool camera2dWideEnabled)
{
    setWindowTitle("Control Panel");

    m_avatarTab = new AvatarTab(m_desireSet);
    m_speechTab = new SpeechTab(m_nodeHandle, m_desireSet);
    m_gestureTab = new GestureTab(m_nodeHandle, m_desireSet);
    m_behaviorsTab = new BehaviorsTab(m_desireSet, camera2dWideEnabled);
    m_ledTab = new LedTab(m_nodeHandle, m_desireSet);
    m_perceptionsTab = new PerceptionsTab(m_nodeHandle, m_desireSet, camera2dWideEnabled);

    m_tabWidget = new QTabWidget;
    m_tabWidget->addTab(m_avatarTab, "Avatar");
    m_tabWidget->addTab(m_speechTab, "Speech");
    m_tabWidget->addTab(m_gestureTab, "Gesture");
    m_tabWidget->addTab(m_behaviorsTab, "Behaviors");
    m_tabWidget->addTab(m_ledTab, "LEDs");
    m_tabWidget->addTab(m_perceptionsTab, "Perceptions");

    m_volumeSlider = new QSlider;
    m_volumeSlider->setOrientation(Qt::Horizontal);
    m_volumeSlider->setMinimum(AUDIO_POWER_AMPLIFIER_MIN_VOLUME);
    m_volumeSlider->setMaximum(AUDIO_POWER_AMPLIFIER_MAX_VOLUME);
    m_volumeSlider->setValue(AUDIO_POWER_AMPLIFIER_DEFAULT_VOLUME);
    connect(m_volumeSlider, &QAbstractSlider::valueChanged, this, &ControlPanel::onVolumeChanged);

    m_batteryLevel = new QLCDNumber;
    m_batteryLevel->display("0");

    auto hlayout = new QHBoxLayout;
    hlayout->addWidget(new QLabel("Volume:"));
    hlayout->addWidget(m_volumeSlider);
    hlayout->addWidget(new QLabel("Battery (%):"));
    hlayout->addWidget(m_batteryLevel);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_tabWidget);
    globalLayout->addLayout(hlayout);

    setLayout(globalLayout);
}
