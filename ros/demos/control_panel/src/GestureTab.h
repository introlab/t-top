#ifndef CONTROL_PANEL_GESTURE_TAB_H
#define CONTROL_PANEL_GESTURE_TAB_H

#include <QWidget>
#include <QPushButton>

class GestureTab : public QWidget
{
    Q_OBJECT

public:
    explicit GestureTab(QWidget* parent = nullptr);

private slots:
    void onGestureButtonToggled(const QString& name, bool checked);

private:
    void createUi();

    // UI members
    QPushButton* m_yesButton;
    QPushButton* m_noButton;
    QPushButton* m_maybeButton;
    QPushButton* m_originAllButton;
    QPushButton* m_originHeadButton;
    QPushButton* m_originTorsoButton;
};

#endif
