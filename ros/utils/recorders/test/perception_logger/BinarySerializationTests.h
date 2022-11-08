#ifndef BINARY_SERIALIZATION_TESTS_H
#define BINARY_SERIALIZATION_TESTS_H

#include <perception_logger/BinarySerialization.h>

template<class T>
T nativeToLittleEndian(const T& v)
{
    if (isLittleEndian())
    {
        return v;
    }
    else
    {
        return switchEndianness(v);
    }
}

template<class T>
T littleEndianToNative(const T& v)
{
    if (isLittleEndian())
    {
        return v;
    }
    else
    {
        return switchEndianness(v);
    }
}

#endif
