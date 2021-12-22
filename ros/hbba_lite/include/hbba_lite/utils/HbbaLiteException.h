#ifndef HBBA_LITE_UTILS_HBBA_LITE_EXCEPTION_H
#define HBBA_LITE_UTILS_HBBA_LITE_EXCEPTION_H

#include <stdexcept>

class HbbaLiteException : public std::runtime_error
{
public:
    HbbaLiteException(const std::string& message);
};

#endif
