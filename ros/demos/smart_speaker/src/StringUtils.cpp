#include "StringUtils.h"

#include <sstream>
#include <algorithm>

using namespace std;

string mergeStrings(const vector<string>& values, const string& separator)
{
    stringstream ss;
    for (size_t i = 0; i < values.size(); i++)
    {
        ss << values[i];
        if (i < values.size() - 1)
        {
            ss << separator;
        }
    }

    return ss.str();
}

string mergeNames(const vector<string>& values, const string& andWord)
{
    stringstream ss;
    for (size_t i = 0; i < values.size(); i++)
    {
        ss << values[i];
        if (i == values.size() - 2)
        {
            ss << " " << andWord << " ";
        }
        else if (i < values.size() - 1)
        {
            ss << ", ";
        }
    }

    return ss.str();
}

// Inspired by https://stackoverflow.com/questions/49201654/splitting-a-string-with-multiple-delimiters-in-c/49201798
vector<string> splitStrings(const string& str, const string& delimiters)
{
    vector<string> values;

    size_t start = 0;
    size_t end = 0;

    while ((start = str.find_first_not_of(delimiters, end)) != string::npos)
    {
        end = str.find_first_of(delimiters, start + 1);
        if (end == string::npos)
        {
            values.push_back(str.substr(start));
        }
        else
        {
            values.push_back(str.substr(start, end - start));
        }
    }

    return values;
}

string toUpperString(const string& str)
{
    string upperStr(str);
    transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
    return upperStr;
}

string toLowerString(const string& str)
{
    string upperStr(str);
    transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::tolower);
    return upperStr;
}

string trimString(const string& str)
{
    static const string SPACES = " \t\n\r";
    size_t begin = str.find_first_not_of(SPACES);
    size_t end = str.find_last_not_of(SPACES);

    if (begin == string::npos || end == string::npos)
    {
        return "";
    }
    else
    {
        return str.substr(begin, end - begin + 1);
    }
}
