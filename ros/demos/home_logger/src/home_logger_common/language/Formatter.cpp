#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

using namespace std;

unique_ptr<Formatter> Formatter::m_instance = nullptr;

Formatter::Formatter(Language language) : m_language(language)
{
    if (StringResources::language() != language)
    {
        throw runtime_error("StringResources::language() must be the same as Formatter::language()");
    }

    if (language == Language::ENGLISH)
    {
        m_locale = locale("C");
    }
    else if (language == Language::FRENCH)
    {
        m_locale = locale("fr_CA.UTF8");
    }
    else
    {
        throw runtime_error("Invalid language");
    }

    m_weekDayNames = StringResources::getVector("week_day_names");
    if (m_weekDayNames.size() != 7)
    {
        throw runtime_error("week_day_names must contain 7 values");
    }

    m_monthNames = StringResources::getVector("month_names");
    if (m_monthNames.size() != 12)
    {
        throw runtime_error("month_names must contain 12 values");
    }
}

void Formatter::initialize(Language language)
{
    m_instance = unique_ptr<Formatter>(new Formatter(language));
}

void Formatter::clear()
{
    m_instance = nullptr;
}
