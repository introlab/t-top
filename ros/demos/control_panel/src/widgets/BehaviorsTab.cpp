#include "BehaviorsTab.h"
#include "../QtUtils.h"
#include "../DesireUtils.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

BehaviorsTab::BehaviorsTab(shared_ptr<DesireSet> desireSet, QWidget* parent)
    : QWidget(parent),
      m_desireSet(std::move(desireSet))
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
    onButtonToggled(
        checked,
        [&]()
        {
            uncheckOtherButtons(m_nearestFaceFollowingButton);

            auto transaction = m_desireSet->beginTransaction();
            removeAllMovementDesires(*m_desireSet);

            auto desire = make_unique<NearestFaceFollowingDesire>();
            m_desireId = static_cast<qint64>(desire->id());
            m_desireSet->addDesire(std::move(desire));
        });
}

void BehaviorsTab::onSpecificFaceFollowingButtonToggled(bool checked)
{
    onButtonToggled(
        checked,
        [&]()
        {
            uncheckOtherButtons(m_specificFaceFollowingButton);

            auto transaction = m_desireSet->beginTransaction();
            removeAllMovementDesires(*m_desireSet);

            auto desire = make_unique<SpecificFaceFollowingDesire>(m_personNameLineEdit->text().toStdString());
            m_desireId = static_cast<qint64>(desire->id());
            m_desireSet->addDesire(std::move(desire));
        });
}

void BehaviorsTab::onSoundFollowingButtonToggled(bool checked)
{
    onButtonToggled(
        checked,
        [&]()
        {
            uncheckOtherButtons(m_soundFollowingButton);

            auto transaction = m_desireSet->beginTransaction();
            removeAllMovementDesires(*m_desireSet);

            auto desire = make_unique<SoundFollowingDesire>();
            m_desireId = static_cast<qint64>(desire->id());
            m_desireSet->addDesire(std::move(desire));
        });
}

void BehaviorsTab::onDanceButtonToggled(bool checked)
{
    onButtonToggled(
        checked,
        [&]()
        {
            uncheckOtherButtons(m_danceButton);

            auto transaction = m_desireSet->beginTransaction();
            removeAllMovementDesires(*m_desireSet);

            auto desire = make_unique<DanceDesire>();
            m_desireId = static_cast<qint64>(desire->id());
            m_desireSet->addDesire(std::move(desire));
        });
}

void BehaviorsTab::onExploreButtonToggled(bool checked)
{
    onButtonToggled(
        checked,
        [&]()
        {
            uncheckOtherButtons(m_exploreAllButton);

            auto transaction = m_desireSet->beginTransaction();
            removeAllMovementDesires(*m_desireSet);

            auto desire = make_unique<ExploreDesire>();
            m_desireId = static_cast<qint64>(desire->id());
            m_desireSet->addDesire(std::move(desire));
        });
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

    m_danceButton = new QPushButton("Dance");
    m_danceButton->setCheckable(true);
    connect(m_danceButton, &QPushButton::toggled, this, &BehaviorsTab::onDanceButtonToggled);

    m_exploreAllButton = new QPushButton("Explore");
    m_exploreAllButton->setCheckable(true);
    connect(m_exploreAllButton, &QPushButton::toggled, this, &BehaviorsTab::onExploreButtonToggled);

    m_personNameLineEdit = new QLineEdit();
    auto personNameLayout = new QHBoxLayout;
    personNameLayout->addWidget(new QLabel("Person Name :"));
    personNameLayout->addWidget(m_personNameLineEdit);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_nearestFaceFollowingButton);
    globalLayout->addWidget(m_specificFaceFollowingButton);
    globalLayout->addWidget(m_soundFollowingButton);
    globalLayout->addWidget(m_danceButton);
    globalLayout->addWidget(m_exploreAllButton);
    globalLayout->addSpacing(20);
    globalLayout->addLayout(personNameLayout);
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
    if (m_exploreAllButton != current)
    {
        m_exploreAllButton->setChecked(false);
    }
}
