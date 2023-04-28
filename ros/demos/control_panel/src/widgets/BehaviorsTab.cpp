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
    m_desireSet->addObserver(this);
}

BehaviorsTab::~BehaviorsTab()
{
    m_desireSet->removeObserver(this);
}

void BehaviorsTab::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    invokeLater(
        [=]()
        {
            if (m_desireId.isValid() && !m_desireSet->contains(m_desireId.toULongLong()))
            {
                m_desireId.clear();
                uncheckOtherButtons(nullptr);
            }
        });
}

void BehaviorsTab::onNearestFaceFollowingButtonToggled(bool checked)
{
    onButtonToggled<NearestFaceFollowingDesire>(checked, m_nearestFaceFollowingButton);
}

void BehaviorsTab::onSpecificFaceFollowingButtonToggled(bool checked)
{
    onButtonToggled<SpecificFaceFollowingDesire>(
        checked,
        m_specificFaceFollowingButton,
        m_personNameLineEdit->text().toStdString());
}

void BehaviorsTab::onSoundFollowingButtonToggled(bool checked)
{
    onButtonToggled<SoundFollowingDesire>(checked, m_soundFollowingButton);
}

void BehaviorsTab::onSoundObjectPersonFollowingButtonToggled(bool checked)
{
    onButtonToggled<SoundObjectPersonFollowingDesire>(checked, m_soundObjectPersonFollowingButton);
}

void BehaviorsTab::onDanceButtonToggled(bool checked)
{
    onButtonToggled<DanceDesire>(checked, m_danceButton);
}

void BehaviorsTab::onExploreButtonToggled(bool checked)
{
    onButtonToggled<ExploreDesire>(checked, m_exploreButton);
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
    globalLayout->addStretch();


    setLayout(globalLayout);
}

void BehaviorsTab::uncheckOtherButtons(QPushButton* current)
{
    if (m_nearestFaceFollowingButton != current)
    {
        m_nearestFaceFollowingButton->setChecked(false);
    }
    if (m_specificFaceFollowingButton != current)
    {
        m_specificFaceFollowingButton->setChecked(false);
    }
    if (m_soundFollowingButton != current)
    {
        m_soundFollowingButton->setChecked(false);
    }
    if (m_danceButton != current)
    {
        m_danceButton->setChecked(false);
    }
    if (m_exploreButton != current)
    {
        m_exploreButton->setChecked(false);
    }
}
