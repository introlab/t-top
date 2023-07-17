#ifndef CONTROL_PANEL_BEHAVIORS_TAB_H
#define CONTROL_PANEL_BEHAVIORS_TAB_H

#include "../DesireUtils.h"

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

    bool m_camera2dWideEnabled;

public:
    BehaviorsTab(std::shared_ptr<DesireSet> desireSet, bool camera2dWideEnabled, QWidget* parent = nullptr);
    ~BehaviorsTab() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

private slots:
    void onNearestFaceFollowingButtonToggled(bool checked);
    void onSpecificFaceFollowingButtonToggled(bool checked);
    void onSoundFollowingButtonToggled(bool checked);
    void onSoundObjectPersonFollowingButtonToggled(bool checked);
    void onDanceButtonToggled(bool checked);
    void onExploreButtonToggled(bool checked);

private:
    void createUi();
    void uncheckOtherButtons(QPushButton* current);

    template<class D, class... DesireArgs>
    void onButtonToggled(bool checked, QPushButton* button, DesireArgs... desireArgs);

    // UI members
    QPushButton* m_nearestFaceFollowingButton;
    QPushButton* m_specificFaceFollowingButton;
    QLineEdit* m_personNameLineEdit;

    QPushButton* m_soundFollowingButton;
    QPushButton* m_soundObjectPersonFollowingButton;
    QPushButton* m_danceButton;
    QPushButton* m_exploreButton;
};

template<class D, class... DesireArgs>
void BehaviorsTab::onButtonToggled(bool checked, QPushButton* button, DesireArgs... desireArgs)
{
    if (checked)
    {
        uncheckOtherButtons(button);

        auto transaction = m_desireSet->beginTransaction();
        removeAllMovementDesires(*m_desireSet);
        if (button == m_danceButton)
        {
            removeAllLedDesires(*m_desireSet);
        }

        auto desire = std::make_unique<D>(desireArgs...);
        m_desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (m_desireId.isValid())
    {
        m_desireSet->removeDesire(m_desireId.toULongLong());
        m_desireId.clear();
    }
}

#endif
