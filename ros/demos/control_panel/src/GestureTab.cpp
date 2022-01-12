#include "GestureTab.h"

#include <QDebug>
#include <QVBoxLayout>

GestureTab::GestureTab(QWidget* parent) : QWidget(parent)
{
    createUi();
}

void GestureTab::onGestureButtonToggled(const QString& name, bool checked)
{
    // TODO
    qDebug() << "onGestureButtonClicked - " << name << " - " << checked;
}

void GestureTab::createUi()
{
    m_yesButton = new QPushButton("Yes");
    m_yesButton->setCheckable(true);
    connect(m_yesButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("yes", checked); });

    m_noButton = new QPushButton("No");
    m_noButton->setCheckable(true);
    connect(m_noButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("no", checked); });

    m_maybeButton = new QPushButton("Maybe");
    m_maybeButton->setCheckable(true);
    connect(m_maybeButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("maybe", checked); });

    m_originAllButton = new QPushButton("Origin All");
    m_originAllButton->setCheckable(true);
    connect(m_originAllButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("origin_all", checked); });

    m_originHeadButton = new QPushButton("Origin Head");
    m_originHeadButton->setCheckable(true);
    connect(m_originHeadButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("origin_head", checked); });

    m_originTorsoButton = new QPushButton("Origin Torso");
    m_originTorsoButton->setCheckable(true);
    connect(m_originTorsoButton, &QPushButton::toggled, this, [this](bool checked) { onGestureButtonToggled("origin_torso", checked); });

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_yesButton);
    globalLayout->addWidget(m_noButton);
    globalLayout->addWidget(m_maybeButton);
    globalLayout->addWidget(m_originAllButton);
    globalLayout->addWidget(m_originHeadButton);
    globalLayout->addWidget(m_originTorsoButton);
    globalLayout->addStretch();

    setLayout(globalLayout);
}
