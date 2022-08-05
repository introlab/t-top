#include <home_logger_common/DateTime.h>

#include <ctime>

using namespace std;

constexpr int TM_YEAR_OFFSET = 1900;

Time::Time() : hour(0), minute(0) {}

Time::Time(int hour, int minute) : hour(hour), minute(minute) {}

Time Time::now()
{
    time_t currentTime = time(nullptr);
    tm buffer;
    localtime_r(&currentTime, &buffer);

    return Time(buffer.tm_hour, buffer.tm_min);
}

bool Time::between(const Time& lowerBound, const Time& upperBound)
{
    if (lowerBound <= upperBound)
    {
        return lowerBound <= *this && *this <= upperBound;
    }
    else
    {
        return *this >= lowerBound || *this <= upperBound;
    }
}


Date::Date() : year(0), month(0), day(0) {}

Date::Date(int year, int month, int day) : year(year), month(month), day(day) {}

Date Date::now()
{
    time_t currentTime = time(nullptr);
    tm buffer;
    localtime_r(&currentTime, &buffer);

    return Date(buffer.tm_year + TM_YEAR_OFFSET, buffer.tm_mon, buffer.tm_mday);
}

int Date::weekDay() const
{
    tm buffer = {0, 0, 0, day, month, year - TM_YEAR_OFFSET};
    time_t tmp = mktime(&buffer);
    localtime_r(&tmp, &buffer);
    return buffer.tm_wday;
}


DateTime::DateTime() {}

DateTime::DateTime(Date date, Time time) : date(date), time(time) {}

DateTime::DateTime(int year, int month, int day, int hour, int minute) : date(year, month, day), time(hour, minute) {}

DateTime DateTime::now()
{
    time_t currentTime = std::time(nullptr);
    tm buffer;
    localtime_r(&currentTime, &buffer);

    return DateTime(
        Date(buffer.tm_year + TM_YEAR_OFFSET, buffer.tm_mon, buffer.tm_mday),
        Time(buffer.tm_hour, buffer.tm_min));
}
