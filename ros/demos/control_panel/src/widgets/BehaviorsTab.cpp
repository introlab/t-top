#include "BehaviorsTab.h"
#include "../QtUtils.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

BehaviorsTab::BehaviorsTab(shared_ptr<DesireSet> desireSet, bool camera2dWideEnabled, QWidget* parent)
    : QWidget(parent),
      m_desireSet(std::move(desireSet)),
      m_camera2dWideEnabled(camera2dWideEnabled)
{
    createUi();
}

BehaviorsTab::~BehaviorsTab() {}

void BehaviorsTab::onNearestFaceFollowingButtonToggled(bool checked)
{
    onButtonToggled<NearestFaceFollowingDesire>(checked, m_nearestFaceFollowingButton, m_nearestFaceFollowingDesireId);
}

void BehaviorsTab::onSpecificFaceFollowingButtonToggled(bool checked)
{
    onButtonToggled<SpecificFaceFollowingDesire>(
        checked,
        m_specificFaceFollowingButton,
        m_specificFaceFollowingDesireId,
        m_personNameLineEdit->text().toStdString());
}

void BehaviorsTab::onSoundFollowingButtonToggled(bool checked)
{
    onButtonToggled<SoundFollowingDesire>(checked, m_soundFollowingButton, m_soundFollowingDesireId);
}

void BehaviorsTab::onSoundObjectPersonFollowingButtonToggled(bool checked)
{
    onButtonToggled<SoundObjectPersonFollowingDesire>(
        checked,
        m_soundObjectPersonFollowingButton,
        m_soundObjectPersonFollowingDesireId);
}

void BehaviorsTab::onDanceButtonToggled(bool checked)
{
    onButtonToggled<DanceDesire>(checked, m_danceButton, m_danceDesireId);
}

void BehaviorsTab::onExploreButtonToggled(bool checked)
{
    onButtonToggled<ExploreDesire>(checked, m_exploreButton, m_exploreDesireId);
}

void BehaviorsTab::onTooCloseReactionButtonToggled(bool checked)
{
    onButtonToggled<TooCloseReactionDesire>(checked, m_tooCloseReactionButton, m_tooCloseReactionDesireId);
}

void BehaviorsTab::createUi()
{
    m_nearestFaceFollowingButton = new QPushButton("Nearest Face Following");
    m_nearestFaceFollowingButton->setCheckable(true);
    connect(
        m_nearestFaceFollowingButton,
        &QPushButton::toggled,
        this,
        &BehaviorsTab::onNearestFaceFollowingButtonToggled);

    m_specificFaceFollowingButton = new QPushButton("Specific Face Following");
    m_specificFaceFollowingButton->setCheckable(true);
    connect(
        m_specificFaceFollowingButton,
        &QPushButton::toggled,
        this,
        &BehaviorsTab::onSpecificFaceFollowingButtonToggled);

    m_soundFollowingButton = new QPushButton("Sound Following");
    m_soundFollowingButton->setCheckable(true);
    connect(m_soundFollowingButton, &QPushButton::toggled, this, &BehaviorsTab::onSoundFollowingButtonToggled);

    if (m_camera2dWideEnabled)
    {
        m_soundObjectPersonFollowingButton = new QPushButton("Sound Object Person Following");
        m_soundObjectPersonFollowingButton->setCheckable(true);
        connect(
            m_soundObjectPersonFollowingButton,
            &QPushButton::toggled,
            this,
            &BehaviorsTab::onSoundObjectPersonFollowingButtonToggled);
    }
    else
    {
        m_soundObjectPersonFollowingButton = nullptr;
    }

    m_danceButton = new QPushButton("Dance");
    m_danceButton->setCheckable(true);
    connect(m_danceButton, &QPushButton::toggled, this, &BehaviorsTab::onDanceButtonToggled);

    m_exploreButton = new QPushButton("Explore");
    m_exploreButton->setCheckable(true);
    connect(m_exploreButton, &QPushButton::toggled, this, &BehaviorsTab::onExploreButtonToggled);

    m_tooCloseReactionButton = new QPushButton("Too Close Reaction");
    m_tooCloseReactionButton->setCheckable(true);
    connect(m_tooCloseReactionButton, &QPushButton::toggled, this, &BehaviorsTab::onTooCloseReactionButtonToggled);

    m_personNameLineEdit = new QLineEdit();
    auto personNameLayout = new QHBoxLayout;
    personNameLayout->addWidget(new QLabel("Person Name :"));
    personNameLayout->addWidget(m_personNameLineEdit);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_nearestFaceFollowingButton);
    globalLayout->addWidget(m_specificFaceFollowingButton);
    globalLayout->addLayout(personNameLayout);
    globalLayout->addSpacing(20);
    globalLayout->addWidget(m_soundFollowingButton);
    if (m_soundObjectPersonFollowingButton != nullptr)
    {
        globalLayout->addWidget(m_soundObjectPersonFollowingButton);
    }
    globalLayout->addWidget(m_danceButton);
    globalLayout->addWidget(m_exploreButton);
    globalLayout->addWidget(m_tooCloseReactionButton);
    globalLayout->addStretch();


    setLayout(globalLayout);
}
