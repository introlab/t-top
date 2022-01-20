#include "BehaviorsTab.h"
#include "../QtUtils.h"
#include "../DesireUtils.h"

#include <QVBoxLayout>

#include <t_top/hbba_lite/Desires.h>

using namespace std;

BehaviorsTab::BehaviorsTab(shared_ptr<DesireSet> desireSet, QWidget* parent) :
        QWidget(parent), m_desireSet(std::move(desireSet))
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
    invokeLater([=]()
    {
        if (m_desireId.isValid() && !m_desireSet->contains(m_desireId.toULongLong()))
        {
            m_desireId.clear();
            m_faceFollowingButton->setChecked(false);
            m_soundFollowingButton->setChecked(false);
            m_danceButton->setChecked(false);
            m_exploreAllButton->setChecked(false);
        }
    });
}

void BehaviorsTab::onFaceFollowingButtonToggled(bool checked)
{
    if (checked)
    {
        m_soundFollowingButton->setChecked(false);
        m_danceButton->setChecked(false);
        m_exploreAllButton->setChecked(false);

        auto transaction = m_desireSet->beginTransaction();
        removeAllMovementDesires(*m_desireSet);

        auto desire = make_unique<FaceFollowingDesire>();
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

void BehaviorsTab::onSoundFollowingButtonToggled(bool checked)
{
    if (checked)
    {
        m_faceFollowingButton->setChecked(false);
        m_danceButton->setChecked(false);
        m_exploreAllButton->setChecked(false);

        auto transaction = m_desireSet->beginTransaction();
        removeAllMovementDesires(*m_desireSet);

        auto desire = make_unique<SoundFollowingDesire>();
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

void BehaviorsTab::onDanceButtonToggled(bool checked)
{
    if (checked)
    {
        m_faceFollowingButton->setChecked(false);
        m_soundFollowingButton->setChecked(false);
        m_exploreAllButton->setChecked(false);

        auto transaction = m_desireSet->beginTransaction();
        removeAllMovementDesires(*m_desireSet);

        auto desire = make_unique<DanceDesire>();
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

void BehaviorsTab::onExploreButtonToggled(bool checked)
{
    if (checked)
    {
        m_faceFollowingButton->setChecked(false);
        m_soundFollowingButton->setChecked(false);
        m_danceButton->setChecked(false);

        auto transaction = m_desireSet->beginTransaction();
        removeAllMovementDesires(*m_desireSet);

        auto desire = make_unique<ExploreDesire>();
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

void BehaviorsTab::createUi()
{
    m_faceFollowingButton = new QPushButton("Face Following");
    m_faceFollowingButton->setCheckable(true);
    connect(m_faceFollowingButton, &QPushButton::toggled, this, &BehaviorsTab::onFaceFollowingButtonToggled);

    m_soundFollowingButton = new QPushButton("Sound Following");
    m_soundFollowingButton->setCheckable(true);
    connect(m_soundFollowingButton, &QPushButton::toggled, this, &BehaviorsTab::onSoundFollowingButtonToggled);

    m_danceButton = new QPushButton("Dance");
    m_danceButton->setCheckable(true);
    connect(m_danceButton, &QPushButton::toggled, this, &BehaviorsTab::onDanceButtonToggled);

    m_exploreAllButton = new QPushButton("Explore");
    m_exploreAllButton->setCheckable(true);
    connect(m_exploreAllButton, &QPushButton::toggled, this, &BehaviorsTab::onExploreButtonToggled);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_faceFollowingButton);
    globalLayout->addWidget(m_soundFollowingButton);
    globalLayout->addWidget(m_danceButton);
    globalLayout->addWidget(m_exploreAllButton);
    globalLayout->addStretch();


    setLayout(globalLayout);
}
