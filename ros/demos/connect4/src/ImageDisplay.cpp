#include "ImageDisplay.h"

#include <QPainter>

#include <algorithm>

using namespace std;

ImageDisplay::ImageDisplay(QWidget* parent) : QWidget(parent) {}

void ImageDisplay::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    painter.fillRect(rect(), QBrush(Qt::black));

    if (m_image.width() <= 0 || m_image.height() <= 0)
    {
        return;
    }

    float scale =
        min(static_cast<float>(width()) / static_cast<float>(m_image.width()),
            static_cast<float>(height()) / static_cast<float>(m_image.height()));
    int scaledWidth = static_cast<int>(scale * m_image.width());
    int scaledHeight = static_cast<int>(scale * m_image.height());
    int offsetX = max(0, (width() - scaledWidth) / 2);
    int offsetY = max(0, (height() - scaledHeight) / 2);

    painter.drawImage(QRect(offsetX, offsetY, scaledWidth, scaledHeight), m_image);
}
