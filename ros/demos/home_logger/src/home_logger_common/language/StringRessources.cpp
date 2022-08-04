#include <home_logger_common/language/StringRessources.h>

using namespace std;

unique_ptr<StringRessources> StringRessources::m_instance = nullptr;

StringRessources::StringRessources(Properties properties, Language language)
    : m_properties(move(properties)),
      m_language(language)
{
}

void StringRessources::loadFromFile(const string& path, Language language)
{
    m_instance = unique_ptr<StringRessources>(new StringRessources(Properties(path), language));
}

void StringRessources::loadFromMap(unordered_map<string, string> properties, Language language)
{
    m_instance = unique_ptr<StringRessources>(new StringRessources(Properties(move(properties)), language));
}

void StringRessources::clear()
{
    m_instance = nullptr;
}

string StringRessources::StringRessources::getValue(const string& key)
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_properties.get<string>(key);
}

vector<string> StringRessources::getVector(const string& key)
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_properties.get<vector<string>>(key);
}

Language StringRessources::language()
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_language;
}
