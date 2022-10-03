#ifndef HOME_LOGGER_COMMON_DATE_TIME_H
#define HOME_LOGGER_COMMON_DATE_TIME_H

struct Time
{
    int hour;  // [0, 23]
    int minute;  // [0, 59]

    Time();
    Time(int hour, int minute);

    bool between(const Time& lowerBound, const Time& upperBound);

    static Time now();

    bool operator==(const Time& other) const;
    bool operator!=(const Time& other) const;
    bool operator<(const Time& other) const;
    bool operator<=(const Time& other) const;
    bool operator>(const Time& other) const;
    bool operator>=(const Time& other) const;
};

inline bool Time::operator==(const Time& other) const
{
    return hour == other.hour && minute == other.minute;
}

inline bool Time::operator!=(const Time& other) const
{
    return !(*this == other);
}

inline bool Time::operator<(const Time& other) const
{
    if (hour == other.hour)
    {
        return minute < other.minute;
    }
    else
    {
        return hour < other.hour;
    }
}

inline bool Time::operator<=(const Time& other) const
{
    if (hour == other.hour)
    {
        return minute <= other.minute;
    }
    else
    {
        return hour <= other.hour;
    }
}

inline bool Time::operator>(const Time& other) const
{
    return !(*this <= other);
}

inline bool Time::operator>=(const Time& other) const
{
    return !(*this < other);
}


struct Date
{
    int year;  // ]-inf, inf[
    int month;  // [0, 11]
    int day;  // [1, 31]

    Date();
    Date(int year, int month, int day);

    static Date now();

    int weekDay() const;

    bool operator==(const Date& other) const;
    bool operator!=(const Date& other) const;
    bool operator<(const Date& other) const;
    bool operator<=(const Date& other) const;
    bool operator>(const Date& other) const;
    bool operator>=(const Date& other) const;
};

inline bool Date::operator==(const Date& other) const
{
    return year == other.year && month == other.month && day == other.day;
}

inline bool Date::operator!=(const Date& other) const
{
    return !(*this == other);
}

inline bool Date::operator<(const Date& other) const
{
    if (year == other.year && month == other.month)
    {
        return day < other.day;
    }
    else if (year == other.year)
    {
        return month < other.month;
    }
    else
    {
        return year < other.year;
    }
}

inline bool Date::operator<=(const Date& other) const
{
    if (year == other.year && month == other.month)
    {
        return day <= other.day;
    }
    else if (year == other.year)
    {
        return month <= other.month;
    }
    else
    {
        return year <= other.year;
    }
}

inline bool Date::operator>(const Date& other) const
{
    return !(*this <= other);
}

inline bool Date::operator>=(const Date& other) const
{
    return !(*this < other);
}


struct DateTime
{
    Date date;
    Time time;

    DateTime();
    DateTime(Date date, Time time);
    DateTime(int year, int month, int day, int hour, int minute);

    static DateTime now();

    bool operator==(const DateTime& other) const;
    bool operator!=(const DateTime& other) const;
    bool operator<(const DateTime& other) const;
    bool operator<=(const DateTime& other) const;
    bool operator>(const DateTime& other) const;
    bool operator>=(const DateTime& other) const;
};

inline bool DateTime::operator==(const DateTime& other) const
{
    return date == other.date && time == other.time;
}

inline bool DateTime::operator!=(const DateTime& other) const
{
    return !(*this == other);
}

inline bool DateTime::operator<(const DateTime& other) const
{
    if (date == other.date)
    {
        return time < other.time;
    }
    else
    {
        return date < other.date;
    }
}

inline bool DateTime::operator<=(const DateTime& other) const
{
    if (date == other.date)
    {
        return time <= other.time;
    }
    else
    {
        return date <= other.date;
    }
}

inline bool DateTime::operator>(const DateTime& other) const
{
    return !(*this <= other);
}

inline bool DateTime::operator>=(const DateTime& other) const
{
    return !(*this < other);
}

#endif
