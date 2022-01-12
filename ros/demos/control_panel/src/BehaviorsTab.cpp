#include "BehaviorsTab.h"

#include <QDebug>
#include <QVBoxLayout>

BehaviorsTab::BehaviorsTab(QWidget* parent) : QWidget(parent)
{
    createUi();
}

void BehaviorsTab::onFaceFollowingButtonToggled(bool checked)
{
    // TODO
    qDebug() << "onFaceFollowingButtonToggled - " << checked;
}

void BehaviorsTab::onSoundFollowingButtonToggled(bool checked)
{
    // TODO
    qDebug() << "onSoundFollowingButtonToggled - " << checked;
}

void BehaviorsTab::onDanceButtonToggled(bool checked)
{
    // TODO
    qDebug() << "onDanceButtonToggled - " << checked;
}

void BehaviorsTab::onExploreButtonToggled(bool checked)
{
    // TODO
    qDebug() << "onExploreButtonToggled - " << checked;
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
