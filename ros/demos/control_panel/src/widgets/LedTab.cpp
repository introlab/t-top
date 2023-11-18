#include "LedTab.h"
#include "../QtUtils.h"
#include "../DesireUtils.h"

#include <QVBoxLayout>
#include <QGroupBox>

#include <t_top_hbba_lite/Desires.h>

using namespace std;


daemon_ros_client::LedColor color(uint8_t r, uint8_t g, uint8_t b)
{
    daemon_ros_client::LedColor c;
    c.red = r;
    c.green = g;
    c.blue = b;
    return c;
}


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

void LedTab::onConstantAnimationButtonToggled(bool checked)
{
    if (checked)
    {
        auto transaction = m_desireSet->beginTransaction();
        removeAllLedDesires(*m_desireSet);
        uncheckOtherButtons(m_constantAnimationButton);

        auto desire =
            make_unique<LedAnimationDesire>("constant", vector<daemon_ros_client::LedColor>{color(255, 255, 255)});
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
    }
}

void LedTab::onRotatingSinAnimationButtonToggled(bool checked)
{
    if (checked)
    {
        auto transaction = m_desireSet->beginTransaction();
        removeAllLedDesires(*m_desireSet);
        uncheckOtherButtons(m_rotatingSinAnimationButton);

        auto desire =
            make_unique<LedAnimationDesire>("rotating_sin", vector<daemon_ros_client::LedColor>{color(0, 255, 0)});
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
    }
}

void LedTab::onRandomAnimationButtonToggled(bool checked)
{
    if (checked)
    {
        auto transaction = m_desireSet->beginTransaction();
        removeAllLedDesires(*m_desireSet);
        uncheckOtherButtons(m_randomAnimationButton);

        auto desire = make_unique<LedAnimationDesire>("random");
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
    m_joyEmotionButton = new QPushButton("Joy");
    m_joyEmotionButton->setCheckable(true);
    connect(
        m_joyEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_joyEmotionButton, checked, "joy"); });

    m_trustEmotionButton = new QPushButton("Trust");
    m_trustEmotionButton->setCheckable(true);
    connect(
        m_trustEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_trustEmotionButton, checked, "trust"); });

    m_sadnessEmotionButton = new QPushButton("Sadness");
    m_sadnessEmotionButton->setCheckable(true);
    connect(
        m_sadnessEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_sadnessEmotionButton, checked, "sadness"); });

    m_fearEmotionButton = new QPushButton("Fear");
    m_fearEmotionButton->setCheckable(true);
    connect(
        m_fearEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_fearEmotionButton, checked, "fear"); });

    m_angerEmotionButton = new QPushButton("Anger");
    m_angerEmotionButton->setCheckable(true);
    connect(
        m_angerEmotionButton,
        &QPushButton::toggled,
        this,
        [this](bool checked) { onLedEmotionButtonToggled(m_angerEmotionButton, checked, "anger"); });

    auto vLayout = new QVBoxLayout;
    vLayout->addWidget(m_joyEmotionButton);
    vLayout->addWidget(m_trustEmotionButton);
    vLayout->addWidget(m_sadnessEmotionButton);
    vLayout->addWidget(m_fearEmotionButton);
    vLayout->addWidget(m_angerEmotionButton);

    auto emotionGroupBox = new QGroupBox("Emotion");
    emotionGroupBox->setLayout(vLayout);


    m_constantAnimationButton = new QPushButton("Constant");
    m_constantAnimationButton->setCheckable(true);
    connect(m_constantAnimationButton, &QPushButton::toggled, this, &LedTab::onConstantAnimationButtonToggled);

    m_rotatingSinAnimationButton = new QPushButton("Rotating Sin");
    m_rotatingSinAnimationButton->setCheckable(true);
    connect(m_rotatingSinAnimationButton, &QPushButton::toggled, this, &LedTab::onRotatingSinAnimationButtonToggled);

    m_randomAnimationButton = new QPushButton("Random");
    m_randomAnimationButton->setCheckable(true);
    connect(m_randomAnimationButton, &QPushButton::toggled, this, &LedTab::onRandomAnimationButtonToggled);

    vLayout = new QVBoxLayout;
    vLayout->addWidget(m_constantAnimationButton);
    vLayout->addWidget(m_rotatingSinAnimationButton);
    vLayout->addWidget(m_randomAnimationButton);

    auto animationGroupBox = new QGroupBox("Animation");
    animationGroupBox->setLayout(vLayout);


    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(emotionGroupBox);
    globalLayout->addWidget(animationGroupBox);
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
    if (m_constantAnimationButton != current)
    {
        m_constantAnimationButton->setChecked(false);
    }
    if (m_rotatingSinAnimationButton != current)
    {
        m_rotatingSinAnimationButton->setChecked(false);
    }
    if (m_randomAnimationButton != current)
    {
        m_randomAnimationButton->setChecked(false);
    }
}
