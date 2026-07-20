#ifndef CONTROL_PANEL_SPEECH_TAB_H
#define CONTROL_PANEL_SPEECH_TAB_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <rclcpp/rclcpp.hpp>

#include <perception_msgs/msg/transcript.hpp>
#include <audio_utils_msgs/msg/voice_activity.hpp>
#include <audio_utils_msgs/msg/complete_utterance.hpp>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class SpeechTab : public QWidget, public DesireSetObserver
{
    Q_OBJECT

    rclcpp::Node::SharedPtr m_node;

    rclcpp::Subscription<perception_msgs::msg::Transcript>::SharedPtr m_speechToTextSubscriber;
    rclcpp::Subscription<audio_utils_msgs::msg::VoiceActivity>::SharedPtr m_vadSubscriber;
    rclcpp::Subscription<audio_utils_msgs::msg::CompleteUtterance>::SharedPtr m_eouSubscriber;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_talkDesireId;
    QVariant m_speechToTextDesireId;
    QVariant m_vadDesireId;
    QVariant m_EouDesireId;

public:
    SpeechTab(rclcpp::Node::SharedPtr node, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
    ~SpeechTab() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

private slots:
    void onTalkButtonClicked();
    void onListenButtonToggled(bool checked);
    void onVadButtonToggled(bool checked);
    void onEouButtonToggled(bool checked);

private:
    void speechToTextSubscriberCallback(const perception_msgs::msg::Transcript::SharedPtr msg);
    void vadSubscriberCallback(const audio_utils_msgs::msg::VoiceActivity::SharedPtr msg);
    void eouSubscriberCallback(const audio_utils_msgs::msg::CompleteUtterance::SharedPtr msg);

    void createUi();

    // UI members
    QTextEdit* m_textToSayTextEdit;
    QPushButton* m_talkButton;
    QPushButton* m_listenButton;
    QTextEdit* m_listenedTextTextEdit;
    QPushButton* m_vadButton;
    QLineEdit* m_vadLineEdit;
    QPushButton* m_eouButton;
    QLineEdit* m_eouLineEdit;
};

#endif
