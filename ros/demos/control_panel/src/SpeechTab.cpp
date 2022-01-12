#include "SpeechTab.h"

#include <QDebug>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>


SpeechTab::SpeechTab(QWidget* parent) : QWidget(parent)
{
    createUi();
}

void SpeechTab::onTalkButtonClicked()
{
    // TODO
    qDebug() << "onTalkButtonClicked";
}

void SpeechTab::onListenButtonToggled(bool checked)
{
    // TODO
    qDebug() << "onListenButtonClicked - " << checked;
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
