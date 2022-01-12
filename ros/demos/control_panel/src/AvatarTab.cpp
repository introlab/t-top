#include "AvatarTab.h"

#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>


constexpr const char* URL = "http://localhost:8080/face";
constexpr int RELOAD_INTERVAL_MS = 10000;

AvatarTab::AvatarTab(QWidget* parent) : QWidget(parent)
{
    createUi();
}

void AvatarTab::onAvatarViewLoadFinished(bool ok)
{
    if (!ok)
    {
        QTimer::singleShot(RELOAD_INTERVAL_MS, this, &AvatarTab::reloadAvatarView);
    }
    else
    {
        onAnimationChanged(m_animationComboBox->currentText());
    }
}

void AvatarTab::onAnimationChanged(const QString& animation)
{
    // TODO
    qDebug() << "onAnimationChanged - " << animation;
}

void AvatarTab::reloadAvatarView()
{
    m_avatarView->reload();
}

void AvatarTab::createUi()
{
    m_avatarView = new QWebView;
    m_avatarView->setUrl(QUrl(URL));
    connect(m_avatarView, &QWebView::loadFinished, this, &AvatarTab::onAvatarViewLoadFinished);

    m_refreshButton = new QPushButton("Refresh");
    connect(m_refreshButton, &QPushButton::clicked, this, &AvatarTab::reloadAvatarView);

    m_animationComboBox = new QComboBox;
    m_animationComboBox->addItem("normal");
    m_animationComboBox->addItem("sleep");
    m_animationComboBox->addItem("blink");
    m_animationComboBox->addItem("wink_left");
    m_animationComboBox->addItem("wink_right");
    m_animationComboBox->addItem("awe");
    m_animationComboBox->addItem("skeptic");
    m_animationComboBox->addItem("angry");
    m_animationComboBox->addItem("sad");
    m_animationComboBox->addItem("disgust");
    m_animationComboBox->addItem("fear");
    m_animationComboBox->addItem("happy");
    connect(m_animationComboBox, &QComboBox::currentTextChanged, this, &AvatarTab::onAnimationChanged);

    auto animationLayout = new QHBoxLayout;
    animationLayout->addWidget(new QLabel("Animation : "));
    animationLayout->addWidget(m_animationComboBox);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_avatarView);
    globalLayout->addWidget(m_refreshButton);
    globalLayout->addLayout(animationLayout);

    setLayout(globalLayout);
}
