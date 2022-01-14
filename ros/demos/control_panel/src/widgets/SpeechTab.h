#ifndef CONTROL_PANEL_SPEECH_TAB_H
#define CONTROL_PANEL_SPEECH_TAB_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QVariant>

#include <ros/ros.h>
#include <talk/Done.h>
#include <std_msgs/String.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class SpeechTab : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;
    ros::Subscriber m_talkDoneSubscriber;
    ros::Subscriber m_speechToTextSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_talkDesireId;
    QVariant m_listenDesireId;

public:
    SpeechTab(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private slots:
    void onTalkButtonClicked();
    void onListenButtonToggled(bool checked);

private:
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
    void speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg);

    void createUi();

    // UI members
    QTextEdit* m_textToSayTextEdit;
    QPushButton* m_talkButton;
    QPushButton* m_listenButton;
    QTextEdit* m_listenedTextTextEdit;
};

#endif
