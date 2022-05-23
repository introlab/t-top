#ifndef CONTROL_PANEL_BEHAVIORS_TAB_H
#define CONTROL_PANEL_BEHAVIORS_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
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
    void onNearestFaceFollowingButtonToggled(bool checked);
    void onSpecificFaceFollowingButtonToggled(bool checked);
    void onSoundFollowingButtonToggled(bool checked);
    void onDanceButtonToggled(bool checked);
    void onExploreButtonToggled(bool checked);

private:
    void createUi();
    void uncheckOtherButtons(QPushButton* current);

    template<class F>
    void onButtonToggled(bool checked, F f);

    // UI members
    QPushButton* m_nearestFaceFollowingButton;
    QLineEdit* m_personNameLineEdit;
    QPushButton* m_specificFaceFollowingButton;
    QPushButton* m_soundFollowingButton;
    QPushButton* m_danceButton;
    QPushButton* m_exploreAllButton;
};

template<class F>
void BehaviorsTab::onButtonToggled(bool checked, F f)
{
    if (checked)
    {
        f();
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

#endif
