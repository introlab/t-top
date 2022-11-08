#ifndef RECORDERS_PERCEPTION_LOGGER_BINARY_SERIALIZATION_H
#define RECORDERS_PERCEPTION_LOGGER_BINARY_SERIALIZATION_H

#include <cstdint>
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
struct always_false : std::false_type
{
};

template<class T>
inline constexpr bool always_false_v = always_false<T>::value;

template<class T>
T switchEndianness(const T& v)
{
    static_assert(always_false_v<T>, "Not supported type");
}

template<>
inline uint32_t switchEndianness(const uint32_t& v)
{
    union
    {
        uint32_t intValue;
        uint8_t bytes[4];
    } unionValue = {v};

    std::swap(unionValue.bytes[0], unionValue.bytes[3]);
    std::swap(unionValue.bytes[1], unionValue.bytes[2]);

    return unionValue.intValue;
}

template<>
inline float switchEndianness(const float& v)
{
    union
    {
        float floatValue;
        uint8_t bytes[4];
    } unionValue = {v};

    std::swap(unionValue.bytes[0], unionValue.bytes[3]);
    std::swap(unionValue.bytes[1], unionValue.bytes[2]);

    return unionValue.floatValue;
}


template<>
inline uint64_t switchEndianness(const uint64_t& v)
{
    union
    {
        uint64_t intValue;
        uint8_t bytes[8];
    } unionValue = {v};

    std::swap(unionValue.bytes[0], unionValue.bytes[7]);
    std::swap(unionValue.bytes[1], unionValue.bytes[6]);
    std::swap(unionValue.bytes[2], unionValue.bytes[5]);
    std::swap(unionValue.bytes[3], unionValue.bytes[4]);

    return unionValue.intValue;
}

template<>
inline double switchEndianness(const double& v)
{
    union
    {
        double doubleValue;
        uint8_t bytes[8];
    } unionValue = {v};

    std::swap(unionValue.bytes[0], unionValue.bytes[7]);
    std::swap(unionValue.bytes[1], unionValue.bytes[6]);
    std::swap(unionValue.bytes[2], unionValue.bytes[5]);
    std::swap(unionValue.bytes[3], unionValue.bytes[4]);

    return unionValue.doubleValue;
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
    friend struct BinarySerializer;
};


template<class T>
struct is_value_type : std::false_type
{
};

template<class T>
inline constexpr bool is_value_type_v = is_value_type<T>::value;


template<class T>
struct BinarySerializer
{
    static Bytes serialize(const T& v)
    {
        static_assert(
            std::is_integral_v<T> || std::is_floating_point_v<T> || is_value_type_v<T>,
            "Not supported type: T must inherit ValueType or be a builtin value type");

        if (isLittleEndian())
        {
            return Bytes(reinterpret_cast<const std::byte*>(&v), sizeof(T));
        }

        Bytes bytes(sizeof(T));
        T* ptr = reinterpret_cast<T*>(bytes.m_data);
        *ptr = switchEndianness(v);
        return bytes;
    }

    static Bytes serialize(T&&) = delete;  // Disallow temporaries
};


template<class T>
struct BinarySerializer<std::vector<T>>
{
    static Bytes serialize(const std::vector<T>& v)
    {
        static_assert(
            std::is_integral_v<T> || std::is_floating_point_v<T> || is_value_type_v<T>,
            "Not supported type: T must be a value type or be a builtin value type");

        if (isLittleEndian())
        {
            return Bytes(reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(T));
        }

        Bytes bytes(v.size() * sizeof(T));
        T* ptr = reinterpret_cast<T*>(bytes.m_data);
        for (size_t i = 0; i < v.size(); i++)
        {
            ptr[i] = switchEndianness(v[i]);
        }
        return bytes;
    }

    static Bytes serialize(std::vector<T>&&) = delete;  // Disallow temporaries
};

#endif
