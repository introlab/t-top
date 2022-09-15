#include <home_logger_common/language/StringResources.h>

using namespace std;

unique_ptr<StringResources> StringResources::m_instance = nullptr;

StringResources::StringResources(Properties properties, Language language)
    : m_properties(move(properties)),
      m_language(language)
{
}

void StringResources::loadFromFile(const string& path, Language language)
{
    m_instance = unique_ptr<StringResources>(new StringResources(Properties(path), language));
}

void StringResources::loadFromMap(unordered_map<string, string> properties, Language language)
{
    m_instance = unique_ptr<StringResources>(new StringResources(Properties(move(properties)), language));
}

void StringResources::clear()
{
    m_instance = nullptr;
}

string StringResources::StringResources::getValue(const string& key)
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_properties.get<string>(key);
}

vector<string> StringResources::getVector(const string& key)
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_properties.get<vector<string>>(key);
}

Language StringResources::language()
{
    if (m_instance == nullptr)
    {
        throw runtime_error("The string ressources are not initialized.");
    }

    return m_instance->m_language;
}
