#include "SpeechTab.h"
#include "../QtUtils.h"

#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

SpeechTab::SpeechTab(ros::NodeHandle& nodeHandle, shared_ptr<DesireSet> desireSet, QWidget* parent)
    : QWidget(parent),
      m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet))
{
    createUi();

    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1, &SpeechTab::talkDoneSubscriberCallback, this);
    m_speechToTextSubscriber =
        nodeHandle.subscribe("speech_to_text/transcript", 1, &SpeechTab::speechToTextSubscriberCallback, this);
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

void SpeechTab::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    invokeLater(
        [=]()
        {
            if (m_talkDesireId.isValid() && m_talkDesireId.toULongLong() == msg->id)
            {
                m_desireSet->removeDesire(m_talkDesireId.toULongLong());
                m_talkDesireId.clear();

                m_talkButton->setEnabled(true);
            }
        });
}

void SpeechTab::speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg)
{
    invokeLater([=]() { m_listenedTextTextEdit->append(QString::fromStdString(msg->data)); });
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

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(new QLabel("Text to say :"));
    globalLayout->addWidget(m_textToSayTextEdit);
    globalLayout->addWidget(m_talkButton);
    globalLayout->addWidget(m_listenButton);
    globalLayout->addWidget(new QLabel("Listened Text :"));
    globalLayout->addWidget(m_listenedTextTextEdit);

    setLayout(globalLayout);
}
