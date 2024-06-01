#include "SpeechTab.h"
#include "../QtUtils.h"

#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

SpeechTab::SpeechTab(rclcpp::Node::SharedPtr node, shared_ptr<DesireSet> desireSet, QWidget* parent)
    : QWidget(parent),
      m_node(std::move(node)),
      m_desireSet(std::move(desireSet))
{
    createUi();
    m_desireSet->addObserver(this);

    m_speechToTextSubscriber =
        m_node->create_subscription<speech_to_text::msg::Transcript>("speech_to_text/transcript", 1, [this](const speech_to_text::msg::Transcript::SharedPtr msg){ speechToTextSubscriberCallback(msg); });
    m_vadSubscriber = m_node->create_subscription<audio_utils::msg::VoiceActivity>("voice_activity", 1, [this](const audio_utils::msg::VoiceActivity::SharedPtr msg) { vadSubscriberCallback(msg); });
}

SpeechTab::~SpeechTab()
{
    m_desireSet->removeObserver(this);
}

void SpeechTab::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    invokeLater(
        [=]()
        {
            if (m_talkDesireId.isValid() && !m_desireSet->contains(m_talkDesireId.toULongLong()))
            {
                m_talkDesireId.clear();
                m_talkButton->setEnabled(true);
            }
        });
}

void SpeechTab::onTalkButtonClicked()
{
    m_talkButton->setEnabled(false);

    auto desire = make_unique<TalkDesire>(m_textToSayTextEdit->toPlainText().toStdString());
    m_talkDesireId = static_cast<qint64>(desire->id());
    m_desireSet->addDesire(std::move(desire));
}

void SpeechTab::onListenButtonToggled(bool checked)
{
    if (checked)
    {
        auto desire = make_unique<SpeechToTextDesire>();
        m_speechToTextDesireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_speechToTextDesireId.isValid())
    {
        m_desireSet->removeDesire(m_speechToTextDesireId.toULongLong());
        m_speechToTextDesireId.clear();
    }
}

void SpeechTab::onVadButtonToggled(bool checked)
{
    if (checked)
    {
        auto desire = make_unique<VadDesire>();
        m_vadDesireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_vadDesireId.isValid())
    {
        m_desireSet->removeDesire(m_vadDesireId.toULongLong());
        m_vadDesireId.clear();
        m_vadLineEdit->setText("");
    }
}

void SpeechTab::speechToTextSubscriberCallback(const speech_to_text::msg::Transcript::SharedPtr msg)
{
    invokeLater([=]() { m_listenedTextTextEdit->append(QString::fromStdString(msg->text)); });
}

void SpeechTab::vadSubscriberCallback(const audio_utils::msg::VoiceActivity::SharedPtr msg)
{
    invokeLater(
        [=]()
        {
            if (!m_vadDesireId.isValid())
            {
                return;
            }

            if (msg->is_voice)
            {
                m_vadLineEdit->setText("Voice");
            }
            else
            {
                m_vadLineEdit->setText("Not voice");
            }
        });
}

void SpeechTab::createUi()
{
    m_textToSayTextEdit = new QTextEdit;
    m_textToSayTextEdit->setAcceptRichText(false);

    m_talkButton = new QPushButton("Talk");
    connect(m_talkButton, &QPushButton::clicked, this, &SpeechTab::onTalkButtonClicked);

    m_listenButton = new QPushButton("Listen");
    m_listenButton->setCheckable(true);
    connect(m_listenButton, &QPushButton::toggled, this, &SpeechTab::onListenButtonToggled);

    m_listenedTextTextEdit = new QTextEdit;
    m_listenedTextTextEdit->setAcceptRichText(false);
    m_listenedTextTextEdit->setReadOnly(true);

    m_vadButton = new QPushButton("VAD");
    m_vadButton->setCheckable(true);
    connect(m_vadButton, &QPushButton::toggled, this, &SpeechTab::onVadButtonToggled);

    m_vadLineEdit = new QLineEdit;
    m_vadLineEdit->setReadOnly(true);

    auto vadLayout = new QHBoxLayout;
    vadLayout->addWidget(m_vadButton, 1);
    vadLayout->addWidget(m_vadLineEdit, 1);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(new QLabel("Text to say :"));
    globalLayout->addWidget(m_textToSayTextEdit);
    globalLayout->addWidget(m_talkButton);
    globalLayout->addWidget(m_listenButton);
    globalLayout->addWidget(new QLabel("Listened Text :"));
    globalLayout->addWidget(m_listenedTextTextEdit);
    globalLayout->addLayout(vadLayout);

    setLayout(globalLayout);
}
