#ifndef CONTROL_PANEL_BEHAVIORS_TAB_H
#define CONTROL_PANEL_BEHAVIORS_TAB_H

#include <QWidget>
#include <QPushButton>

class BehaviorsTab : public QWidget
{
    Q_OBJECT

public:
    explicit BehaviorsTab(QWidget* parent = nullptr);

private slots:
    void onFaceFollowingButtonToggled(bool checked);
    void onSoundFollowingButtonToggled(bool checked);
    void onDanceButtonToggled(bool checked);
    void onExploreButtonToggled(bool checked);

private:
    void createUi();

    // UI members
    QPushButton* m_faceFollowingButton;
    QPushButton* m_soundFollowingButton;
    QPushButton* m_danceButton;
    QPushButton* m_exploreAllButton;
};

#endif
