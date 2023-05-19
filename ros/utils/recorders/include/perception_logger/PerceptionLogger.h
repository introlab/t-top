#ifndef RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H

#include <perception_logger/BinarySerialization.h>

#include <ros/ros.h>

struct __attribute__((packed)) Position
{
    double x;
    double y;
    double z;
};

template<>
struct is_value_type<Position> : std::true_type
{
};

template<>
inline std::array<std::byte, sizeof(Position)> toLittleEndianBytes(const Position& v)
{
    std::array<std::byte, sizeof(double)> xBytes = toLittleEndianBytes(v.x);
    std::array<std::byte, sizeof(double)> yBytes = toLittleEndianBytes(v.y);
    std::array<std::byte, sizeof(double)> zBytes = toLittleEndianBytes(v.z);

    return joinArrays(xBytes, yBytes, zBytes);
}

template<>
inline Position fromLittleEndianBytes(const std::array<std::byte, sizeof(Position)>& bytes)
{
    auto [xBytes, yBytes, zBytes] = splitArray<double, double, double>(bytes);

    return Position{
        fromLittleEndianBytes<double>(xBytes),
        fromLittleEndianBytes<double>(yBytes),
        fromLittleEndianBytes<double>(zBytes)};
}


struct __attribute__((packed)) ImagePosition
{
    double x;
    double y;
};

template<>
struct is_value_type<ImagePosition> : std::true_type
{
};

template<>
inline std::array<std::byte, sizeof(ImagePosition)> toLittleEndianBytes(const ImagePosition& v)
{
    std::array<std::byte, sizeof(ImagePosition)> bytes;
    std::array<std::byte, sizeof(double)> xBytes = toLittleEndianBytes(v.x);
    std::array<std::byte, sizeof(double)> yBytes = toLittleEndianBytes(v.y);
    std::memcpy(bytes.data(), xBytes.data(), sizeof(double));
    std::memcpy(bytes.data() + sizeof(double), yBytes.data(), sizeof(double));

    return bytes;
}

template<>
inline ImagePosition fromLittleEndianBytes(const std::array<std::byte, sizeof(ImagePosition)>& bytes)
{
    std::array<std::byte, sizeof(double)> xBytes;
    std::array<std::byte, sizeof(double)> yBytes;
    std::memcpy(xBytes.data(), bytes.data(), sizeof(double));
    std::memcpy(yBytes.data(), bytes.data() + sizeof(double), sizeof(double));

    return ImagePosition{fromLittleEndianBytes<double>(xBytes), fromLittleEndianBytes<double>(yBytes)};
}


struct __attribute__((packed)) BoundingBox
{
    ImagePosition center;
    double width;
    double height;
};


struct __attribute__((packed)) Direction
{
    double x;
    double y;
    double z;
};

template<>
struct is_value_type<Direction> : std::true_type
{
};

template<>
inline std::array<std::byte, sizeof(Direction)> toLittleEndianBytes(const Direction& v)
{
    std::array<std::byte, sizeof(Position)> bytes;
    std::array<std::byte, sizeof(double)> xBytes = toLittleEndianBytes(v.x);
    std::array<std::byte, sizeof(double)> yBytes = toLittleEndianBytes(v.y);
    std::array<std::byte, sizeof(double)> zBytes = toLittleEndianBytes(v.z);
    std::memcpy(bytes.data(), xBytes.data(), sizeof(double));
    std::memcpy(bytes.data() + sizeof(double), yBytes.data(), sizeof(double));
    std::memcpy(bytes.data() + 2 * sizeof(double), zBytes.data(), sizeof(double));

    return bytes;
}

template<>
inline Direction fromLittleEndianBytes(const std::array<std::byte, sizeof(Direction)>& bytes)
{
    std::array<std::byte, sizeof(double)> xBytes;
    std::array<std::byte, sizeof(double)> yBytes;
    std::array<std::byte, sizeof(double)> zBytes;
    std::memcpy(xBytes.data(), bytes.data(), sizeof(double));
    std::memcpy(yBytes.data(), bytes.data() + sizeof(double), sizeof(double));
    std::memcpy(zBytes.data(), bytes.data() + 2 * sizeof(double), sizeof(double));

    return Direction{
        fromLittleEndianBytes<double>(xBytes),
        fromLittleEndianBytes<double>(yBytes),
        fromLittleEndianBytes<double>(zBytes)};
}

struct Timestamp
{
    int64_t unixEpochMs;

    explicit Timestamp(int64_t unixEpochMs) : unixEpochMs(unixEpochMs) {}

    Timestamp(const ros::Time& time)
        : unixEpochMs{static_cast<int64_t>(time.sec) * 1'000 + static_cast<int64_t>(time.nsec) / 1'000'000}
    {
    }
};

#endif
