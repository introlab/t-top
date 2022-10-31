#ifndef RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_PERCEPTION_LOGGER_H

#include <ros/ros.h>

struct __attribute__((packed)) Position
{
    double x;
    double y;
    double z;
};

struct __attribute__((packed)) ImagePosition
{
    double x;
    double y;
};

struct BoundingBox
{
    ImagePosition centre;
    double width;
    double height;
};


struct __attribute__((packed)) Direction
{
    double x;
    double y;
    double z;
};

struct __attribute__((packed)) Timestamp
{
    int64_t unixEpoch;

    Timestamp(int64_t unixEpoch) : unixEpoch(unixEpoch) {}

    Timestamp(const ros::Time& time) : unixEpoch(time.sec) {}
};

#endif
