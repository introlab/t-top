#ifndef CONTROL_PANEL_GESTURE_TAB_H
#define CONTROL_PANEL_GESTURE_TAB_H

#include <QWidget>
#include <QPushButton>
#include <QVariant>

#include <hbba_lite/core/DesireSet.h>

#include <memory>
#include <utility>

class GestureTab : public QWidget, public DesireSetObserver
{
    Q_OBJECT

    std::shared_ptr<DesireSet> m_desireSet;
    QVariant m_gestureDesireId;

public:
    GestureTab(std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);
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
    QPushButton* m_slowOriginHeadButton;
    QPushButton* m_originTorsoButton;
    QPushButton* m_thinkingButton;
    QPushButton* m_sadButton;
};

#endif
