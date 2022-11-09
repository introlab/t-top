#ifndef RECORDERS_PERCEPTION_LOGGER_BINARY_SERIALIZATION_H
#define RECORDERS_PERCEPTION_LOGGER_BINARY_SERIALIZATION_H

#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <vector>

inline constexpr bool isLittleEndian()
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return true;
#else
    return false;
#endif
}


template<class T>
struct is_value_type : std::false_type
{
};

template<class T>
inline constexpr bool is_value_type_v = is_value_type<T>::value;


template<class T>
std::array<std::byte, sizeof(T)> toLittleEndianBytes(const T& v)
{
    static_assert(
        std::is_integral_v<T> || std::is_floating_point_v<T>,
        "Not supported type: T must be a builtin value type");

    std::array<std::byte, sizeof(T)> bytes;
    std::memcpy(bytes.data(), &v, sizeof(T));

    if constexpr (!isLittleEndian())
    {
        for (size_t i = 0; i < sizeof(T) / 2; i++)
        {
            std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
        }
    }

    return bytes;
}

template<class T>
T fromLittleEndianBytes(const std::array<std::byte, sizeof(T)>& bytes)
{
    static_assert(
        std::is_integral_v<T> || std::is_floating_point_v<T>,
        "Not supported type: T must be a builtin value type");

    T v;
    std::memcpy(&v, bytes.data(), sizeof(T));

    if constexpr (!isLittleEndian())
    {
        std::byte* ptr = reinterpret_cast<std::byte*>(&v);
        for (size_t i = 0; i < sizeof(T) / 2; i++)
        {
            std::swap(ptr[i], ptr[sizeof(T) - 1 - i]);
        }
    }

    return v;
}


class Bytes
{
    bool m_owned;
    std::byte* m_data;
    size_t m_size;

public:
    explicit Bytes(size_t size);
    Bytes(const std::byte* data, size_t size);
    Bytes(const Bytes&) = delete;
    Bytes(Bytes&& other);
    ~Bytes();

    bool owned() const { return m_owned; }

    const std::byte* data() const { return m_data; }
    size_t size() const { return m_size; }

    Bytes& operator=(const Bytes& other) = delete;
    Bytes& operator=(Bytes&& other);

    template<class T>
    friend Bytes serializeToBytesNoCopy(const T& v);

    template<class T>
    friend Bytes serializeToBytesNoCopy(const std::vector<T>& v);
};


template<class T>
Bytes serializeToBytesNoCopy(const T& v)
{
    static_assert(
        std::is_integral_v<T> || std::is_floating_point_v<T> || is_value_type_v<T>,
        "Not supported type: T must inherit ValueType or be a builtin value type");

    if constexpr (isLittleEndian())
    {
        return Bytes(reinterpret_cast<const std::byte*>(&v), sizeof(T));
    }

    Bytes bytes(sizeof(T));
    auto littleEndianBytes = toLittleEndianBytes(v);
    memcpy(bytes.m_data, littleEndianBytes.data(), sizeof(T));
    return bytes;
}

template<class T>
Bytes serializeToBytesNoCopy(T& v)
{
    return serializeToBytesNoCopy(const_cast<const T&>(v));
};

/**
 * To prevent the compiler to call "Bytes serializeToBytesNoCopy(T&&)" when a non-const type is provided.
 */
template<class T>
Bytes serializeToBytesNoCopy(T&&) = delete;  // Disallow temporaries

template<class T>
Bytes serializeToBytesNoCopy(const std::vector<T>& v)
{
    static_assert(
        std::is_integral_v<T> || std::is_floating_point_v<T> || is_value_type_v<T>,
        "Not supported type: T must be a value type or be a builtin value type");

    if constexpr (isLittleEndian())
    {
        return Bytes(reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(T));
    }

    Bytes bytes(v.size() * sizeof(T));
    T* ptr = reinterpret_cast<T*>(bytes.m_data);
    for (size_t i = 0; i < v.size(); i++)
    {
        auto littleEndianBytes = toLittleEndianBytes(v[i]);
        memcpy(ptr + i, littleEndianBytes.data(), sizeof(T));
    }
    return bytes;
}

/**
 * To prevent the compiler to call "Bytes serializeToBytesNoCopy(std::vector<T>&&)" when a non-const vector is provided.
 */
template<class T>
Bytes serializeToBytesNoCopy(std::vector<T>& v)
{
    return serializeToBytesNoCopy(const_cast<const std::vector<T>&>(v));
};

template<class T>
Bytes serializeToBytesNoCopy(std::vector<T>&&) = delete;  // Disallow temporaries

#endif
