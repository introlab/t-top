#ifndef CONTROL_PANEL_BEHAVIORS_TAB_H
#define CONTROL_PANEL_BEHAVIORS_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QVariant>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class BehaviorsTab : public QWidget
{
    Q_OBJECT

    std::shared_ptr<DesireSet> m_desireSet;

    QVariant m_nearestFaceFollowingDesireId;
    QVariant m_specificFaceFollowingDesireId;
    QVariant m_soundFollowingDesireId;
    QVariant m_soundObjectPersonFollowingDesireId;
    QVariant m_danceDesireId;
    QVariant m_exploreDesireId;
    QVariant m_tooCloseReactionDesireId;

    bool m_camera2dWideEnabled;

public:
    BehaviorsTab(std::shared_ptr<DesireSet> desireSet, bool camera2dWideEnabled, QWidget* parent = nullptr);
    ~BehaviorsTab() override;

private slots:
    void onNearestFaceFollowingButtonToggled(bool checked);
    void onSpecificFaceFollowingButtonToggled(bool checked);
    void onSoundFollowingButtonToggled(bool checked);
    void onSoundObjectPersonFollowingButtonToggled(bool checked);
    void onDanceButtonToggled(bool checked);
    void onExploreButtonToggled(bool checked);
    void onTooCloseReactionButtonToggled(bool checked);

private:
    void createUi();

    template<class D, class... DesireArgs>
    void onButtonToggled(bool checked, QPushButton* button, QVariant& desireId, DesireArgs... desireArgs);

    // UI members
    QPushButton* m_nearestFaceFollowingButton;
    QPushButton* m_specificFaceFollowingButton;
    QLineEdit* m_personNameLineEdit;

    QPushButton* m_soundFollowingButton;
    QPushButton* m_soundObjectPersonFollowingButton;
    QPushButton* m_danceButton;
    QPushButton* m_exploreButton;
    QPushButton* m_tooCloseReactionButton;
};

template<class D, class... DesireArgs>
void BehaviorsTab::onButtonToggled(bool checked, QPushButton* button, QVariant& desireId, DesireArgs... desireArgs)
{
    if (checked)
    {
        auto desire = std::make_unique<D>(desireArgs...);
        desireId = static_cast<qint64>(desire->id());
        m_desireSet->addDesire(std::move(desire));
    }
    else if (desireId.isValid())
    {
        m_desireSet->removeDesire(desireId.toULongLong());
        desireId.clear();
    }
}

#endif
