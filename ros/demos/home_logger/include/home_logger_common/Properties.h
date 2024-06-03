#ifndef HOME_LOGGER_COMMON_PROPERTIES_H
#define HOME_LOGGER_COMMON_PROPERTIES_H

#include <home_logger_common/StringUtils.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <istream>
#include <sstream>
#include <stdexcept>

class Properties
{
    std::unordered_map<std::string, std::string> m_properties;

public:
    Properties(std::unordered_map<std::string, std::string> properties);
    explicit Properties(const std::string& filename);
    virtual ~Properties();

    template<class T>
    T get(const std::string& key) const;

    std::unordered_set<std::string> keys() const;

private:
    void parse(std::istream& stream);
    void parseLine(const std::string& line);
};

template<class T>
struct ValueParser
{
    static T parse(const std::string& key, const std::string& valueStr)
    {
        std::istringstream ss(valueStr);

        T value;
        ss >> value;

        if (ss.fail())
        {
            throw std::runtime_error("Properties : parse failed (" + key + ", " + valueStr + ")");
        }

        return value;
    }
};

template<>
struct ValueParser<bool>
{
    static bool parse(const std::string& key, const std::string& valueStr) { return toLowerString(valueStr) == "true"; }
};

template<>
struct ValueParser<std::string>
{
    static std::string parse(const std::string& key, const std::string& valueStr) { return valueStr; }
};

template<class T>
struct ValueParser<std::vector<T>>
{
    static std::vector<T> parse(const std::string& key, const std::string& valueStr)
    {
        if (valueStr.size() < 2 || valueStr[0] != '[' || valueStr[valueStr.size() - 1] != ']')
        {
            throw std::runtime_error("Properties : parse failed (" + key + ", " + valueStr + ")");
        }

        std::vector<T> values;
        std::string arrayValue;
        std::stringstream arrayValuesStream(valueStr.substr(1, valueStr.size() - 2));
        while (std::getline(arrayValuesStream, arrayValue, ','))
        {
            trim(arrayValue);
            if (arrayValue != "" || !arrayValuesStream.eof())
            {
                values.push_back(ValueParser<T>::parse(key, arrayValue));
            }
        }

        return values;
    }
};

template<class T>
inline T Properties::get(const std::string& key) const
{
    auto it = m_properties.find(key);
    if (it == m_properties.end())
    {
        throw std::runtime_error("Properties : key not found (" + key + ")");
    }

    return ValueParser<T>::parse(key, it->second);
}

#endif
