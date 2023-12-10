#include "GestureTab.h"
#include "../QtUtils.h"

#include <QVBoxLayout>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

GestureTab::GestureTab(ros::NodeHandle& nodeHandle, shared_ptr<DesireSet> desireSet, QWidget* parent)
    : QWidget(parent),
      m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet))
{
    createUi();
    m_desireSet->addObserver(this);
}

GestureTab::~GestureTab()
{
    m_desireSet->removeObserver(this);
}

void GestureTab::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    invokeLater(
        [=]()
        {
            if (m_gestureDesireId.isValid() && !m_desireSet->contains(m_gestureDesireId.toULongLong()))
            {
                m_gestureDesireId.clear();
                setEnabledAllButtons(true);
            }
        });
}

void GestureTab::onGestureButtonClicked(const QString& name)
{
    setEnabledAllButtons(false);

    auto desire = make_unique<GestureDesire>(name.toStdString());
    m_gestureDesireId = static_cast<qint64>(desire->id());
    m_desireSet->addDesire(std::move(desire));
}

void GestureTab::setEnabledAllButtons(bool enabled)
{
    m_yesButton->setEnabled(enabled);
    m_noButton->setEnabled(enabled);
    m_maybeButton->setEnabled(enabled);
    m_originAllButton->setEnabled(enabled);
    m_originHeadButton->setEnabled(enabled);
    m_originTorsoButton->setEnabled(enabled);
}

void GestureTab::createUi()
{
    m_yesButton = new QPushButton("Yes");
    connect(m_yesButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("yes"); });

    m_noButton = new QPushButton("No");
    connect(m_noButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("no"); });

    m_maybeButton = new QPushButton("Maybe");
    connect(m_maybeButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("maybe"); });

    m_originAllButton = new QPushButton("Origin All");
    connect(m_originAllButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("origin_all"); });

    m_originHeadButton = new QPushButton("Origin Head");
    connect(m_originHeadButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("origin_head"); });

    m_originTorsoButton = new QPushButton("Origin Torso");
    connect(m_originTorsoButton, &QPushButton::clicked, this, [this]() { onGestureButtonClicked("origin_torso"); });

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
