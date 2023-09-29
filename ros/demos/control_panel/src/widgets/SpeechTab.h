#ifndef CONTROL_PANEL_SPEECH_TAB_H
#define CONTROL_PANEL_SPEECH_TAB_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <ros/ros.h>
#include <talk/Done.h>
#include <speech_to_text/Transcript.h>
#include <audio_utils/VoiceActivity.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class SpeechTab : public QWidget, public DesireSetObserver
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    ros::Subscriber m_speechToTextSubscriber;
    ros::Subscriber m_vadSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_talkDesireId;
    QVariant m_speechToTextDesireId;
    QVariant m_vadDesireId;

public:
    SpeechTab(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
    ~SpeechTab() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

private slots:
    void onTalkButtonClicked();
    void onListenButtonToggled(bool checked);
    void onVadButtonToggled(bool checked);

private:
    void speechToTextSubscriberCallback(const speech_to_text::Transcript::ConstPtr& msg);
    void vadSubscriberCallback(const audio_utils::VoiceActivity::ConstPtr& msg);

    void createUi();

    // UI members
    QTextEdit* m_textToSayTextEdit;
    QPushButton* m_talkButton;
    QPushButton* m_listenButton;
    QTextEdit* m_listenedTextTextEdit;
    QPushButton* m_vadButton;
    QLineEdit* m_vadLineEdit;
};

#endif
