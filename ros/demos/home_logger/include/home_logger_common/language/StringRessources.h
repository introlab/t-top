#ifndef HOME_LOGGER_COMMON_LANGUAGE_STRING_RESOURCES_H
#define HOME_LOGGER_COMMON_LANGUAGE_STRING_RESOURCES_H

#include <home_logger_common/Properties.h>
#include <home_logger_common/language/Language.h>

#include <hbba_lite/utils/ClassMacros.h>

#include <memory>
#include <unordered_map>
#include <vector>

class StringRessources
{
    static std::unique_ptr<StringRessources> m_instance;

    Properties m_properties;
    Language m_language;

    StringRessources(Properties properties, Language language);

    DECLARE_NOT_COPYABLE(StringRessources);
    DECLARE_NOT_MOVABLE(StringRessources);

public:
    static void loadFromFile(const std::string& path, Language language);
    static void loadFromMap(std::unordered_map<std::string, std::string> properties, Language language);
    static void clear();  // For tests only

    static std::string getValue(const std::string& key);
    static std::vector<std::string> getVector(const std::string& key);
    static Language language();
};

#endif
