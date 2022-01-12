#ifndef CONTROL_PANEL_SPEECH_TAB_H
#define CONTROL_PANEL_SPEECH_TAB_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>

class SpeechTab : public QWidget
{
    Q_OBJECT

public:
    explicit SpeechTab(QWidget* parent = nullptr);

private slots:
    void onTalkButtonClicked();
    void onListenButtonToggled(bool checked);

private:
    void createUi();

    // UI members
    QTextEdit* m_textToSayTextEdit;
    QPushButton* m_talkButton;
    QPushButton* m_listenButton;
    QTextEdit* m_listenedTextTextEdit;
};

#endif
