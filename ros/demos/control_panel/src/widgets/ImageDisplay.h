#ifndef CONTROL_PANEL_IMAGE_DISPLAY_H
#define CONTROL_PANEL_IMAGE_DISPLAY_H

#include <QWidget>
#include <QPushButton>

// Inspired by https://github.com/introlab/opentera-webrtc-ros/blob/main/opentera_webrtc_robot_gui/src/ROSCameraView.h
class ImageDisplay : public QWidget
{
    Q_OBJECT

    QImage m_image;

public:
    explicit ImageDisplay(QWidget* parent = nullptr);

public slots:
    void setImage(const QImage& image);

protected:
    void paintEvent(QPaintEvent* event) override;
};

inline void ImageDisplay::setImage(const QImage& image)
{
    m_image = image.copy();
    update();
}

#endif
