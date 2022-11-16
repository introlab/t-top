#ifndef COMMUNICATION_SERIAL_COMMUNICATION_BUFFER_H
#define COMMUNICATION_SERIAL_COMMUNICATION_BUFFER_H

#include <tl/optional.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <array>

template<class T>
struct enum_min : std::integral_constant<typename std::underlying_type<T>::type, 0>
{
};

template<class T>
struct enum_max : std::integral_constant<typename std::underlying_type<T>::type, 0>
{
};

template<class T>
bool write(uint8_t* data, size_t& writeIndex, size_t maxSize, const T& value)
{
    static_assert(
        std::is_integral<T>::value || std::is_floating_point<T>::value || std::is_enum<T>::value,
        "Not supported type: T must be a value type.");

    if ((maxSize - writeIndex) < sizeof(T))
    {
        return false;
    }

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(data + writeIndex, &value, sizeof(T));
#else
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));


    for (size_t i = 0; i < sizeof(T) / 2; i++)
    {
        std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
    }

    std::memcpy(data + writeIndex, bytes, sizeof(T));
#endif

    writeIndex += sizeof(T);
    return true;
}

inline bool write(uint8_t* data, size_t& writeIndex, size_t maxSize, const uint8_t* newData, size_t newDataSize)
{
    if ((maxSize - writeIndex) < newDataSize)
    {
        return false;
    }

    std::memcpy(data + writeIndex, newData, newDataSize);
    writeIndex += newDataSize;
    return true;
}

template<>
inline bool write(uint8_t* data, size_t& writeIndex, size_t maxSize, const bool& value)
{
    return write(data, writeIndex, maxSize, static_cast<uint8_t>(value));
}

template<class T>
typename std::enable_if<!std::is_enum<T>::value && !std::is_same<T, bool>::value, tl::optional<T>>::type
    read(const uint8_t* data, size_t writeIndex, size_t& readIndex, size_t maxSize)
{
    static_assert(
        std::is_integral<T>::value || std::is_floating_point<T>::value,
        "Not supported type: T must be a value type.");

    if ((writeIndex - readIndex) < sizeof(T))
    {
        return tl::nullopt;
    }

    typename std::remove_const<T>::type value;

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(&value, data + readIndex, sizeof(T));
#else
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, data + readIndex, sizeof(T));


    for (size_t i = 0; i < sizeof(T) / 2; i++)
    {
        std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
    }

    std::memcpy(value, bytes, sizeof(T));
#endif

    readIndex += sizeof(T);
    return value;
}

template<class T>
typename std::enable_if<std::is_same<T, bool>::value, tl::optional<T>>::type
    read(const uint8_t* data, size_t writeIndex, size_t& readIndex, size_t maxSize)
{
    auto value = read<uint8_t>(data, writeIndex, readIndex, maxSize);
    if (!value.has_value())
    {
        return tl::nullopt;
    }

    return value != 0;
}

template<class T>
typename std::enable_if<std::is_enum<T>::value, tl::optional<T>>::type
    read(const uint8_t* data, size_t writeIndex, size_t& readIndex, size_t maxSize)
{
    auto value = read<typename std::underlying_type<T>::type>(data, writeIndex, readIndex, maxSize);
    if (!value.has_value() || *value < enum_min<T>::value || *value > enum_max<T>::value)
    {
        return tl::nullopt;
    }

    return static_cast<T>(*value);
}


template<size_t N>
class SerialCommunicationBuffer;


/**
 * Little-endian buffer view
 */
class SerialCommunicationBufferView
{
    uint8_t* m_data;
    size_t& m_writeIndex;
    size_t& m_readIndex;
    size_t m_maxSize;

public:
    template<size_t N>
    SerialCommunicationBufferView(SerialCommunicationBuffer<N>& buffer);

    size_t maxSize() const;
    size_t sizeToWrite() const;
    size_t sizeToRead() const;

    void clear();
    void moveToBeginning();
    const uint8_t* dataToRead() const;

    template<class T>
    bool write(const T& value);
    bool write(const uint8_t* data, size_t size);

    template<class T>
    tl::optional<T> read() const;
};


/**
 * Little-endian buffer
 */
template<size_t N>
class SerialCommunicationBuffer
{
    uint8_t m_data[N];
    mutable size_t m_writeIndex;
    mutable size_t m_readIndex;

public:
    SerialCommunicationBuffer();

    size_t maxSize() const;
    size_t sizeToWrite() const;
    size_t sizeToRead() const;

    void clear();
    void moveToBeginning();
    const uint8_t* dataToRead() const;

    template<class T>
    bool write(const T& value);
    bool write(const uint8_t* data, size_t size);

    template<class T>
    tl::optional<T> read() const;

    friend class SerialCommunicationBufferView;
};

template<size_t N>
SerialCommunicationBuffer<N>::SerialCommunicationBuffer() : m_writeIndex(0),
                                                            m_readIndex(0)
{
}

template<size_t N>
inline size_t SerialCommunicationBuffer<N>::maxSize() const
{
    return N;
}

template<size_t N>
inline size_t SerialCommunicationBuffer<N>::sizeToWrite() const
{
    return N - m_writeIndex;
}

template<size_t N>
inline size_t SerialCommunicationBuffer<N>::sizeToRead() const
{
    return m_writeIndex - m_readIndex;
}

template<size_t N>
inline void SerialCommunicationBuffer<N>::clear()
{
    m_writeIndex = 0;
    m_readIndex = 0;
}

template<size_t N>
inline void SerialCommunicationBuffer<N>::moveToBeginning()
{
    std::memmove(m_data, m_data + m_readIndex, sizeToRead());
    m_writeIndex -= m_readIndex;
    m_readIndex = 0;
}

template<size_t N>
inline const uint8_t* SerialCommunicationBuffer<N>::dataToRead() const
{
    return m_data + m_readIndex;
}

template<size_t N>
template<class T>
inline bool SerialCommunicationBuffer<N>::write(const T& value)
{
    return ::write(m_data, m_writeIndex, N, value);
}

template<size_t N>
inline bool SerialCommunicationBuffer<N>::write(const uint8_t* data, size_t size)
{
    return ::write(m_data, m_writeIndex, N, data, size);
}

template<size_t N>
template<class T>
inline tl::optional<T> SerialCommunicationBuffer<N>::read() const
{
    return ::read<T>(m_data, m_writeIndex, m_readIndex, N);
}


template<size_t N>
SerialCommunicationBufferView::SerialCommunicationBufferView(SerialCommunicationBuffer<N>& buffer)
    : m_data(buffer.m_data),
      m_writeIndex(buffer.m_writeIndex),
      m_readIndex(buffer.m_readIndex),
      m_maxSize(N)
{
}

inline size_t SerialCommunicationBufferView::maxSize() const
{
    return m_maxSize;
}

inline size_t SerialCommunicationBufferView::sizeToWrite() const
{
    return m_maxSize - m_writeIndex;
}

inline size_t SerialCommunicationBufferView::sizeToRead() const
{
    return m_writeIndex - m_readIndex;
}

inline void SerialCommunicationBufferView::clear()
{
    m_writeIndex = 0;
    m_readIndex = 0;
}

inline void SerialCommunicationBufferView::moveToBeginning()
{
    std::memmove(m_data, m_data + m_readIndex, sizeToRead());
    m_writeIndex -= m_readIndex;
    m_readIndex = 0;
}

inline const uint8_t* SerialCommunicationBufferView::dataToRead() const
{
    return m_data + m_readIndex;
}

template<class T>
inline bool SerialCommunicationBufferView::write(const T& value)
{
    return ::write(m_data, m_writeIndex, m_maxSize, value);
}

inline bool SerialCommunicationBufferView::write(const uint8_t* data, size_t size)
{
    return ::write(m_data, m_writeIndex, m_maxSize, data, size);
}

template<class T>
inline tl::optional<T> SerialCommunicationBufferView::read() const
{
    return ::read<T>(m_data, m_writeIndex, m_readIndex, m_maxSize);
}

#endif
