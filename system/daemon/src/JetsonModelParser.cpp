#include "JetsonModelParser.h"

#include <QFile>

#include <string>
#include <string_view>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>

#include <tl/optional.hpp>


std::string get_jetson_model_name(JetsonModel model)
{
    switch (model)
    {
        case JetsonModel::XAVIER:
            return "Xavier";
        case JetsonModel::ORIN:
            return "Orin";
        case JetsonModel::UNKNOWN:
            return "Unknown";
        default:
            throw std::runtime_error(
                std::string("Unimplemented name for JetsonModel value [") +
                std::to_string(static_cast<std::underlying_type_t<JetsonModel>>(model)) + "]");
    }
}

static tl::optional<std::string> read_file(const std::string& path)
{
    if (!QFile(QString::fromStdString(path)).exists())
    {
        return tl::nullopt;
    }

    std::ifstream file = std::ifstream{path};
    std::string content = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

    return content;
}

static std::vector<std::string> split_string(const std::string& text, char delimiter)
{
    std::stringstream ss;
    std::vector<std::string> out;

    for (auto character : text)
    {
        if (delimiter == character)
        {
            out.push_back(ss.str());
            ss.str("");  // clear
        }
        else
        {
            ss << character;
        }
    }
    if (!ss.str().empty())
    {
        out.push_back(ss.str());
    }
    return out;
}

static tl::optional<std::string> get_p_number(const std::string& text)
{
    auto parts = split_string(text, '/');
    auto dts_part = parts.back();
    auto dts_parts = split_string(dts_part, '-');
    if (dts_parts.size() < 3)
    {
        return tl::nullopt;
    }
    auto p_number = std::string{dts_parts[1]} + "-" + std::string{dts_parts[2]};
    return p_number;
}

static tl::optional<JetsonModel> get_model_from_p_number(const std::string& p_number)
{
    // https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/IN/QuickStart.html
    static const std::map<std::string, JetsonModel> P_NUMBER_MAPPING = {
        {"p3701-0000", JetsonModel::ORIN},
        {"p2888-0001", JetsonModel::XAVIER},
        {"p2888-0003", JetsonModel::XAVIER},
        {"p2888-0005", JetsonModel::XAVIER},
    };

    if (P_NUMBER_MAPPING.count(p_number) == 0)
    {
        return tl::nullopt;
    }

    return P_NUMBER_MAPPING.at(p_number);
}

JetsonModel get_jetson_model()
{
    return read_file("/proc/device-tree/nvidia,dtsfilename")
        .and_then(get_p_number)
        .and_then(get_model_from_p_number)
        .value_or(JetsonModel::UNKNOWN);
}
