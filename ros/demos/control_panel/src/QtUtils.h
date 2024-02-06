#ifndef CONTROL_PANEL_QT_UTILS_H
#define CONTROL_PANEL_QT_UTILS_H

#include <QEvent>
#include <QCoreApplication>

#include <functional>

// Inspired by https://stackoverflow.com/questions/42566421/how-to-queue-lambda-function-into-qts-event-loop
class FunctionEvent : public QEvent
{
    std::function<void()> m_f;

public:
    FunctionEvent(std::function<void()>&& f);
    ~FunctionEvent() override;
};

inline FunctionEvent::FunctionEvent(std::function<void()>&& f) : QEvent(QEvent::None), m_f(move(f)) {}

inline FunctionEvent::~FunctionEvent()
{
    m_f();
}

inline void invokeLater(std::function<void()> f)
{
    if (qApp != nullptr)
    {
        QCoreApplication::postEvent(qApp, new FunctionEvent(move(f)));
    }
}


#endif
