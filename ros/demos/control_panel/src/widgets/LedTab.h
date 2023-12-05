#ifndef CONTROL_PANEL_LED_TAB_H
#define CONTROL_PANEL_LED_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QVariant>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class LedTab : public QWidget
{
    Q_OBJECT

    ros::NodeHandle& m_nodeHandle;

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_ledEmotionDesireId;
    QVariant m_ledAnimationDesireId;

public:
    LedTab(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
    ~LedTab() override;

private slots:
    void onLedEmotionButtonToggled(QPushButton* button, bool checked, const QString& name);
    void onConstantAnimationButtonToggled(bool checked);
    void onRotatingSinAnimationButtonToggled(bool checked);
    void onRandomAnimationButtonToggled(bool checked);

private:
    void createUi();
    void uncheckOtherEmotionButtons(QPushButton* current);
    void uncheckOtherAnimationButtons(QPushButton* current);

    // UI members
    QPushButton* m_joyEmotionButton;
    QPushButton* m_trustEmotionButton;
    QPushButton* m_sadnessEmotionButton;
    QPushButton* m_fearEmotionButton;
    QPushButton* m_angerEmotionButton;

    QPushButton* m_constantAnimationButton;
    QPushButton* m_rotatingSinAnimationButton;
    QPushButton* m_randomAnimationButton;
};

#endif
