#include "ControlPanel.h"
#include "../QtUtils.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

#include <std_msgs/Int8.h>

using namespace std;

constexpr int AUDIO_POWER_AMPLIFIER_MIN_VOLUME = 0;
constexpr int AUDIO_POWER_AMPLIFIER_MAX_VOLUME = 63;
constexpr int AUDIO_POWER_AMPLIFIER_DEFAULT_VOLUME = 24;

ControlPanel::ControlPanel(ros::NodeHandle& nodeHandle, shared_ptr<DesireSet> desireSet, QWidget* parent) :
        QWidget(parent), m_nodeHandle(nodeHandle), m_desireSet(std::move(desireSet))
{
    m_volumePublisher = nodeHandle.advertise<std_msgs::Int8>("opencr/audio_power_amplifier_volume", 1);

    createUi();

    m_batterySubscriber = nodeHandle.subscribe("opencr/state_of_charge_voltage_current", 1,
        &ControlPanel::batterySubscriberCallback, this);
}

void ControlPanel::onVolumeChanged(int volume)
{
    std_msgs::Int8 msg;
    msg.data = volume;
    m_volumePublisher.publish(msg);
}

void ControlPanel::batterySubscriberCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    if (msg->data.size() == 3)
    {
        int battery = static_cast<int>(msg->data[0]);
        invokeLater([this, battery]()
        {
            m_batteryLevel->display(QString::number(battery));
        });
    }
}

void ControlPanel::createUi()
{
    setWindowTitle("Control Panel");

    m_avatarTab = new AvatarTab(m_desireSet);
    m_speechTab = new SpeechTab(m_nodeHandle, m_desireSet);
    m_gestureTab = new GestureTab(m_nodeHandle, m_desireSet);
    m_behaviorsTab = new BehaviorsTab(m_desireSet);
    m_perceptionsTab = new PerceptionsTab(m_nodeHandle, m_desireSet);

    m_tabWidget = new QTabWidget;
    m_tabWidget->addTab(m_avatarTab, "Avatar");
    m_tabWidget->addTab(m_speechTab, "Speech");
    m_tabWidget->addTab(m_gestureTab, "Gesture");
    m_tabWidget->addTab(m_behaviorsTab, "Behaviors");
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
