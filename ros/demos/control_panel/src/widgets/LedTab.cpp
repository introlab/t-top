#include "LedTab.h"
#include "../QtUtils.h"
#include "../DesireUtils.h"

#include <QVBoxLayout>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

LedTab::LedTab(ros::NodeHandle& nodeHandle, shared_ptr<DesireSet> desireSet, QWidget* parent)
    : QWidget(parent),
      m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet))
{
    createUi();
    m_desireSet->addObserver(this);
}

LedTab::~LedTab()
{
    m_desireSet->removeObserver(this);
}

void LedTab::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
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

void LedTab::onLedEmotionButtonToggled(QPushButton* button, bool checked, const QString& name)
{
    if (checked)
    {
        auto transaction = m_desireSet->beginTransaction();
        removeAllLedDesires(*m_desireSet);
        uncheckOtherButtons(button);

        auto desire = make_unique<LedEmotionDesire>(name.toStdString());
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
    }
}

void LedTab::createUi()
{
    m_joyEmotionButton = new QPushButton("Joy Emotion");
    m_joyEmotionButton->setCheckable(true);
    connect(
        m_joyEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_joyEmotionButton, checked, "joy"); });

    m_trustEmotionButton = new QPushButton("Trust Emotion");
    m_trustEmotionButton->setCheckable(true);
    connect(
        m_trustEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_trustEmotionButton, checked, "trust"); });

    m_sadnessEmotionButton = new QPushButton("Sadness Emotion");
    m_sadnessEmotionButton->setCheckable(true);
    connect(
        m_sadnessEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_sadnessEmotionButton, checked, "sadness"); });

    m_fearEmotionButton = new QPushButton("Fear Emotion");
    m_fearEmotionButton->setCheckable(true);
    connect(
        m_fearEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_fearEmotionButton, checked, "fear"); });

    m_angerEmotionButton = new QPushButton("Anger Emotion");
    m_angerEmotionButton->setCheckable(true);
    connect(
        m_angerEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_angerEmotionButton, checked, "anger"); });

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_joyEmotionButton);
    globalLayout->addWidget(m_trustEmotionButton);
    globalLayout->addWidget(m_sadnessEmotionButton);
    globalLayout->addWidget(m_fearEmotionButton);
    globalLayout->addWidget(m_angerEmotionButton);
    globalLayout->addStretch();

    setLayout(globalLayout);
}

void LedTab::uncheckOtherButtons(QPushButton* current)
{
    if (m_joyEmotionButton != current)
    {
        m_joyEmotionButton->setChecked(false);
    }
    if (m_trustEmotionButton != current)
    {
        m_trustEmotionButton->setChecked(false);
    }
    if (m_sadnessEmotionButton != current)
    {
        m_sadnessEmotionButton->setChecked(false);
    }
    if (m_fearEmotionButton != current)
    {
        m_fearEmotionButton->setChecked(false);
    }
    if (m_angerEmotionButton != current)
    {
        m_angerEmotionButton->setChecked(false);
    }
}
