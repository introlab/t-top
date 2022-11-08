#ifndef RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H

#include <perception_logger/BinarySerialization.h>

#include <ros/ros.h>

struct __attribute__((packed)) Position : public ValueType
{
    double x;
    double y;
    double z;

    Position() : x(0.0), y(0.0), z(0.0) {}
    Position(double x, double y, double z) : x(x), y(y), z(z) {}
};

template<>
inline Position switchEndianness(const Position& v)
{
    return Position{switchEndianness(v.x), switchEndianness(v.y), switchEndianness(v.z)};
}


struct __attribute__((packed)) ImagePosition : public ValueType
{
    double x;
    double y;

    ImagePosition() : x(0.0), y(0.0) {}
    ImagePosition(double x, double y) : x(x), y(y) {}
};

template<>
inline ImagePosition switchEndianness(const ImagePosition& v)
{
    return ImagePosition{switchEndianness(v.x), switchEndianness(v.y)};
}

struct BoundingBox
{
    ImagePosition center;
    double width;
    double height;
};


struct __attribute__((packed)) Direction : public ValueType
{
    double x;
    double y;
    double z;

    Direction() : x(0), y(0), z(0) {}
    Direction(double x, double y, double z) : x(x), y(y), z(z) {}
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
