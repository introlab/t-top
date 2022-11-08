#ifndef COMPARISONS_H
#define COMPARISONS_H

#include <perception_logger/PerceptionLogger.h>
#include <gtest/gtest.h>

static bool represents_integer(double d)
{
    return std::floor(d) == std::ceil(d);
}

static bool operator==(const Position& l, const Position& r)
{
    EXPECT_TRUE(represents_integer(l.x));
    EXPECT_TRUE(represents_integer(l.y));
    EXPECT_TRUE(represents_integer(l.z));

    EXPECT_TRUE(represents_integer(r.x));
    EXPECT_TRUE(represents_integer(r.y));
    EXPECT_TRUE(represents_integer(r.z));

    return l.x == r.x && l.y == r.y && l.z == r.z;
}

static bool operator==(const ImagePosition& l, const ImagePosition& r)
{
    EXPECT_TRUE(represents_integer(l.x));
    EXPECT_TRUE(represents_integer(l.y));

    EXPECT_TRUE(represents_integer(r.x));
    EXPECT_TRUE(represents_integer(r.y));

    return l.x == r.x && l.y == r.y;
}

static bool operator==(const BoundingBox& l, const BoundingBox& r)
{
    EXPECT_TRUE(represents_integer(l.width));
    EXPECT_TRUE(represents_integer(l.height));

    EXPECT_TRUE(represents_integer(r.width));
    EXPECT_TRUE(represents_integer(r.height));

    return l.center == r.center && l.width == r.width && l.height == r.height;
}

static bool operator==(const Direction& l, const Direction& r)
{
    EXPECT_TRUE(represents_integer(l.x));
    EXPECT_TRUE(represents_integer(l.y));
    EXPECT_TRUE(represents_integer(l.z));

    EXPECT_TRUE(represents_integer(r.x));
    EXPECT_TRUE(represents_integer(r.y));
    EXPECT_TRUE(represents_integer(r.z));

    return l.x == r.x && l.y == r.y && l.z == r.z;
}

#endif
