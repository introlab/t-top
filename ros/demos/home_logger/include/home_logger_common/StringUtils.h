#ifndef HOME_LOGGER_COMMON_STRING_UTILS_H
#define HOME_LOGGER_COMMON_STRING_UTILS_H

#include <algorithm>
#include <functional>
#include <codecvt>
#include <locale>
#include <string>

inline std::string toUpperString(const std::string& str)
{
    std::locale utf8("en_US.UTF-8");
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    std::wstring upperStr = converter.from_bytes(str);
    std::transform(
        upperStr.begin(),
        upperStr.end(),
        upperStr.begin(),
        [&utf8](wchar_t c) { return std::toupper(c, utf8); });
    return converter.to_bytes(upperStr);
}

inline std::string toLowerString(const std::string& str)
{
    std::locale utf8("en_US.UTF-8");
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    std::wstring lowerStr = converter.from_bytes(str);
    std::transform(
        lowerStr.begin(),
        lowerStr.end(),
        lowerStr.begin(),
        [&utf8](wchar_t c) { return std::tolower(c, utf8); });
    return converter.to_bytes(lowerStr);
}

inline std::string& trimLeft(std::string& str)
{
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return str;
}

inline std::string& trimRight(std::string& str)
{
    str.erase(
        std::find_if(str.rbegin(), str.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
        str.end());
    return str;
}

inline std::string& trim(std::string& str)
{
    return trimLeft(trimRight(str));
}

#endif
