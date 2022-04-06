#include "PsuControlCommandHandler.h"

#include <Arduino.h>

constexpr uint8_t MESSAGE_SIZE_INDEX = 0;
constexpr uint8_t MESSAGE_TYPE_INDEX = 1;

constexpr uint8_t MESSAGE_TYPE_STATUS = 0;

PsuControlCommandHandler::PsuControlCommandHandler() : m_statusCommandHandler(nullptr)
{
    clearMessage();
}

void PsuControlCommandHandler::update()
{
    while (Serial1.available() > 0)
    {
        readOneByte();
    }
}

void PsuControlCommandHandler::setStatusCommandHandler(StatusCommandHandler handler)
{
    m_statusCommandHandler = handler;
}

void PsuControlCommandHandler::readOneByte()
{
    m_buffer[m_bufferIndex] = Serial1.read();
    m_bufferIndex++;

    if (m_bufferIndex >= m_buffer[MESSAGE_SIZE_INDEX])
    {
        handleMessage();
    }
}

void PsuControlCommandHandler::handleMessage()
{
    switch (m_buffer[MESSAGE_TYPE_INDEX])
    {
        case MESSAGE_TYPE_STATUS:
            if (m_statusCommandHandler != nullptr)
            {
                bool isPsuConnected = m_buffer[2];
                float stateOfCharge, current, voltage;

                memcpy(&stateOfCharge, m_buffer + 3, sizeof(float));
                memcpy(&current, m_buffer + 7, sizeof(float));
                memcpy(&voltage, m_buffer + 11, sizeof(float));
                m_statusCommandHandler(isPsuConnected, stateOfCharge, current, voltage);
            }
            break;
    }

    clearMessage();
}

void PsuControlCommandHandler::clearMessage()
{
    memset(m_buffer, 0, UINT8_MAX);
    m_bufferIndex = 0;
}
