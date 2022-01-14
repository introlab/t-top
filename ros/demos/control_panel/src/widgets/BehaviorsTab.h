#ifndef CONTROL_PANEL_BEHAVIORS_TAB_H
#define CONTROL_PANEL_BEHAVIORS_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QVariant>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class BehaviorsTab : public QWidget, public DesireSetObserver
{
    Q_OBJECT

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_desireId;

public:
    BehaviorsTab(std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
    ~BehaviorsTab() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

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
