#include <home_logger_common/language/StringRessources.h>
#include <home_logger_common/language/Formatter.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

void loadFrenchStringResources()
{
    fs::path testFilePath(__FILE__);
    fs::path propertiesFilePath =
        testFilePath.parent_path().parent_path().parent_path() / "resources" / "strings_fr.properties";

    StringRessources::loadFromFile(propertiesFilePath.string(), Language::FRENCH);
    Formatter::initialize(Language::FRENCH);
}

void loadEnglishStringResources()
{
    fs::path testFilePath(__FILE__);
    fs::path propertiesFilePath =
        testFilePath.parent_path().parent_path().parent_path() / "resources" / "strings_en.properties";

    StringRessources::loadFromFile(propertiesFilePath.string(), Language::ENGLISH);
    Formatter::initialize(Language::ENGLISH);
}
