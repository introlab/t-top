#ifndef CONTROL_PANEL_AVATAR_TAB_H
#define CONTROL_PANEL_AVATAR_TAB_H

#include <QWidget>
#include <QWebView>
#include <QPushButton>
#include <QComboBox>

class AvatarTab : public QWidget
{
    Q_OBJECT

public:
    explicit AvatarTab(QWidget* parent = nullptr);

private slots:
    void onAvatarViewLoadFinished(bool ok);
    void onAnimationChanged(const QString& animation);

    void reloadAvatarView();

private:
    void createUi();

    // UI members
    QWebView* m_avatarView;
    QPushButton* m_refreshButton;
    QComboBox* m_animationComboBox;
};

#endif
