#include <home_logger_common/parameters/CommandParametersParser.h>

#include <home_logger_common/DateTime.h>
#include <home_logger_common/StringUtils.h>
#include <home_logger_common/language/StringResources.h>
#include <home_logger_common/language/Formatter.h>

using namespace std;

CommandParametersParser::CommandParametersParser() {}

CommandParametersParser::~CommandParametersParser() {}


bool containsAny(const string& text, const vector<string>& keywords)
{
    for (auto& keyword : keywords)
    {
        if (text.find(keyword) != string::npos)
        {
            return true;
        }
    }
    return false;
}

static optional<int> findInt(const string& text, size_t& startPosition, size_t& endPosition)
{
    constexpr const char* DIGITS = "0123456789";
    startPosition = text.find_first_of(DIGITS);
    if (startPosition == string::npos)
    {
        return nullopt;
    }
    if (startPosition > 0 && text[startPosition - 1] == '-')
    {
        startPosition--;
    }
    endPosition = text.find_first_not_of(DIGITS, startPosition + 1);
    return stoi(text.substr(startPosition, endPosition));
}

optional<int> findInt(const string& text)
{
    size_t startPosition;
    size_t endPosition;
    return findInt(text, startPosition, endPosition);
}

static optional<Time> findTimeHourOnly(const string& lowerCaseText)
{
    optional<int> hour = findInt(lowerCaseText);
    if (!hour.has_value())
    {
        return nullopt;
    }

    bool am = containsAny(lowerCaseText, StringResources::getVector("time.am"));
    bool pm = containsAny(lowerCaseText, StringResources::getVector("time.pm"));
    if ((am && pm) || (!am && !pm) || (am && hour.value() > 11))
    {
        return nullopt;
    }
    if (pm && hour.value() < 12)
    {
        hour.value() += 12;
    }

    return Time(hour.value(), 0);
}

optional<Time> findTime(const string& text)
{
    string lowerCaseText = toLowerString(text);
    size_t separatorPosition = lowerCaseText.find_first_of("h:.");
    if (separatorPosition == string::npos)
    {
        return findTimeHourOnly(lowerCaseText);
    }

    optional<int> hour = findInt(lowerCaseText.substr(0, separatorPosition));
    if (!hour.has_value() || hour.value() < 0 || hour.value() > 23)
    {
        return nullopt;
    }

    optional<int> minute = findInt(lowerCaseText.substr(separatorPosition));
    if (!minute.has_value())
    {
        minute = 0;
    }
    else if (minute.value() < 0 || minute.value() > 59)
    {
        return nullopt;
    }

    bool am = containsAny(lowerCaseText, StringResources::getVector("time.am"));
    bool pm = containsAny(lowerCaseText, StringResources::getVector("time.pm"));
    if ((am && pm) || (am && hour.value() > 11))
    {
        return nullopt;
    }
    if (pm && hour.value() < 12)
    {
        hour.value() += 12;
    }

    return Time(hour.value(), minute.value());
}

static optional<int> findMonth(const string& lowerCaseText, size_t& monthPosition)
{
    const vector<string>& monthNames = Formatter::monthNames();

    optional<int> month;
    for (int i = 0; i < monthNames.size(); i++)
    {
        size_t position = lowerCaseText.find(toLowerString(monthNames[i]));
        bool found = position != string::npos;
        if (found && !month.has_value())
        {
            monthPosition = position;
            month = i;
        }
        else if (found)
        {
            return -1;
        }
    }

    return month;
}

static optional<Date> findDayOnlyDate(const string& text, int defaultYear, int defaultMonth)
{
    optional<int> day = findInt(text);
    if (day.has_value() && 1 <= day.value() && day.value() <= 31)
    {
        return Date(defaultYear, defaultMonth, day.value());
    }

    return nullopt;
}

static optional<int> findDay(const string& lowerCaseText, size_t monthPosition, size_t& dayEndPosition)
{
    size_t dayStartPosition;
    switch (Formatter::language())
    {
        case Language::ENGLISH:
            return findInt(lowerCaseText.substr(monthPosition), dayStartPosition, dayEndPosition);
        case Language::FRENCH:
            return findInt(lowerCaseText.substr(0, monthPosition), dayStartPosition, dayEndPosition);
        default:
            throw runtime_error("Invalid language");
    }
}

static optional<int> findYear(const string& lowerCaseText, size_t monthPosition, size_t dayEndPosition)
{
    switch (Formatter::language())
    {
        case Language::ENGLISH:
            if (dayEndPosition == string::npos)
            {
                return nullopt;
            }
            else
            {
                return findInt(lowerCaseText.substr(dayEndPosition));
            }
        case Language::FRENCH:
            return findInt(lowerCaseText.substr(monthPosition));
        default:
            throw runtime_error("Invalid language");
    }
}

optional<Date> findDate(const string& text, int defaultYear, int defaultMonth)
{
    string lowerCaseText = toLowerString(text);

    if (containsAny(lowerCaseText, StringResources::getVector("date.today")))
    {
        return Date::now();
    }

    size_t monthPosition;
    optional<int> month = findMonth(lowerCaseText, monthPosition);
    if (!month.has_value())
    {
        return findDayOnlyDate(lowerCaseText, defaultYear, defaultMonth);
    }
    else if (month < 0 || month > 11)
    {
        return nullopt;
    }

    size_t dayEndPosition;
    optional<int> day = findDay(lowerCaseText, monthPosition, dayEndPosition);
    if (!day.has_value() || day.value() < 1 || day.value() > 31)
    {
        return nullopt;
    }

    optional<int> year = findYear(lowerCaseText, monthPosition, dayEndPosition);
    if (!year.has_value())
    {
        year = defaultYear;
    }

    return Date(year.value(), month.value(), day.value());
}

optional<int> findWeekDay(const string& text)
{
    string lowerCaseText = toLowerString(text);

    optional<int> weekDay;
    const vector<string>& weekDayNames = Formatter::weekDayNames();

    for (int i = 0; i < weekDayNames.size(); i++)
    {
        bool found = lowerCaseText.find(toLowerString(weekDayNames[i])) != string::npos;
        if (found && !weekDay.has_value())
        {
            weekDay = i;
        }
        else if (found)
        {
            return nullopt;
        }
    }

    return weekDay;
}
