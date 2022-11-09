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


template<class... Pack>
inline constexpr std::size_t total_sizeof_pack = (sizeof(Pack) + ... + 0);

template<std::size_t... Pack>
inline constexpr std::size_t sum_pack = (Pack + ... + 0);


template<std::size_t... Sizes>
std::array<std::byte, sum_pack<Sizes...>> joinArrays(const std::array<std::byte, Sizes>&... arrays)
{
    std::array<std::byte, sum_pack<Sizes...>> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), arrays.size(), result.begin() + index), index += arrays.size()), ...);

    return result;
}

// https://artificial-mind.net/blog/2020/10/31/constexpr-for
template<size_t Start, size_t End, size_t Inc, class F>
constexpr void constexpr_for(F&& f)
{
    if constexpr (Start < End)
    {
        f(std::integral_constant<size_t, Start>());
        constexpr_for<Start + Inc, End, Inc>(f);
    }
}
template<class F, class Tuple>
constexpr void constexpr_for_tuple(F&& f, Tuple&& tuple)
{
    constexpr size_t cnt = std::tuple_size_v<std::decay_t<Tuple>>;

    constexpr_for<size_t{0}, cnt, size_t{1}>([&](auto i) { f(std::get<i.value>(tuple)); });
}


template<class... ArraysTypes, size_t N>
std::tuple<std::array<std::byte, sizeof(ArraysTypes)>...> splitArray(const std::array<std::byte, N>& arr)
{
    static_assert(total_sizeof_pack<ArraysTypes...> == N, "Sum of output sizeofs should match length of input array");

    using TupleType = std::tuple<std::array<std::byte, sizeof(ArraysTypes)>...>;
    TupleType result;
    std::size_t index{};

    constexpr_for_tuple(
        [&index, &arr](auto& v)
        {
            std::copy_n(arr.begin() + index, v.size(), v.begin());
            index += v.size();
        },
        result);

    return result;
}


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

/**
 * To prevent the compiler to call "Bytes serializeToBytesNoCopy(T&&)" when a non-const type is provided.
 */
template<class T>
Bytes serializeToBytesNoCopy(T& v)
{
    return serializeToBytesNoCopy(const_cast<const T&>(v));
};

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
