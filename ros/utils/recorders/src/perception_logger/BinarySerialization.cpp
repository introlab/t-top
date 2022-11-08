#include <perception_logger/BinarySerialization.h>

#include <cstdlib>

using namespace std;

Bytes::Bytes(size_t size) : m_owned(true), m_data(new byte[size]), m_size(size) {}

Bytes::Bytes(const byte* data, size_t size) : m_owned(false), m_data(const_cast<byte*>(data)), m_size(size) {}

Bytes::Bytes(Bytes&& other)
{
    if (this == &other)
    {
        return;
    }

    m_owned = other.m_owned;
    m_data = other.m_data;
    m_size = other.m_size;

    other.m_owned = false;
    other.m_data = nullptr;
    other.m_size = 0;
}

Bytes::~Bytes()
{
    if (m_owned)
    {
        delete[] m_data;
    }
}

Bytes& Bytes::operator=(Bytes&& other)
{
    if (this == &other)
    {
        return *this;
    }

    if (m_owned)
    {
        delete[] m_data;
    }

    m_owned = other.m_owned;
    m_data = other.m_data;
    m_size = other.m_size;

    other.m_owned = false;
    other.m_data = nullptr;
    other.m_size = 0;

    return *this;
}
