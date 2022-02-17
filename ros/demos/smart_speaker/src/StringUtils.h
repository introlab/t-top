#ifndef SMART_SPEAKER_STRING_UTILS_H
#define SMART_SPEAKER_STRING_UTILS_H

#include <string>
#include <vector>

std::string mergeStrings(const std::vector<std::string>& values, const std::string& separator);
std::string mergeNames(const std::vector<std::string>& values, const std::string& andWord);
std::vector<std::string> splitStrings(const std::string& str, const std::string& delimiters);
std::string toUpperString(const std::string& str);
std::string toLowerString(const std::string& str);
std::string trimString(const std::string& str);

#endif
