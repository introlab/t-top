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
inline Position switchEndianness(const Position& v)
{
    return Position{switchEndianness(v.x), switchEndianness(v.y), switchEndianness(v.z)};
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
inline ImagePosition switchEndianness(const ImagePosition& v)
{
    return ImagePosition{switchEndianness(v.x), switchEndianness(v.y)};
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
inline Direction switchEndianness(const Direction& v)
{
    return Direction{switchEndianness(v.x), switchEndianness(v.y), switchEndianness(v.z)};
}

struct Timestamp
{
    int64_t unixEpochMs;

    explicit Timestamp(int64_t unixEpochMs) : unixEpochMs(unixEpochMs) {}

    Timestamp(const ros::Time& time) : unixEpochMs(time.sec * 1'000 + time.nsec / 1'000'000) {}
};

#endif
