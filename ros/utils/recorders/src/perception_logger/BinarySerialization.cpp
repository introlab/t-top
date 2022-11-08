#include <perception_logger/BinarySerialization.h>

#include <cstdlib>

using namespace std;

Bytes::Bytes(size_t size) : m_owned(true), m_data(malloc(size)), m_size(size) {}

Bytes::Bytes(const void* data, size_t size) : m_owned(false), m_data(const_cast<void*>(data)), m_size(size) {}

Bytes::Bytes(Bytes&& other) : m_owned(other.m_owned), m_data(other.m_data), m_size(other.m_size)
{
    other.m_owned = false;
    other.m_data = nullptr;
    other.m_size = 0;
}

Bytes::~Bytes()
{
    if (m_owned)
    {
        free(m_data);
    }
}

Bytes& Bytes::operator=(Bytes&& other)
{
    if (m_owned)
    {
        free(m_data);
    }

    m_owned = other.m_owned;
    m_data = other.m_data;
    m_size = other.m_size;

    other.m_owned = false;
    other.m_data = nullptr;
    other.m_size = 0;

    return *this;
}
