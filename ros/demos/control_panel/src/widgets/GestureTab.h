#ifndef CONTROL_PANEL_GESTURE_TAB_H
#define CONTROL_PANEL_GESTURE_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QVariant>

#include <ros/ros.h>
#include <gesture/Done.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class GestureTab : public QWidget, public DesireSetObserver
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_gestureDesireId;

public:
    GestureTab(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
    ~GestureTab() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

private slots:
    void onGestureButtonClicked(const QString& name);

private:
    void setEnabledAllButtons(bool enabled);

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
