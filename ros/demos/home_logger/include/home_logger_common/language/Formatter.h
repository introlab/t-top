#ifndef HOME_LOGGER_COMMON_LANGUAGE_FORMATTER_H
#define HOME_LOGGER_COMMON_LANGUAGE_FORMATTER_H

#include <home_logger_common/DateTime.h>
#include <home_logger_common/language/Language.h>

#include <hbba_lite/utils/ClassMacros.h>

#include <fmt/core.h>
#include <fmt/format.h>

#include <memory>
#include <locale>
#include <vector>
#include <cmath>

template<>
struct fmt::formatter<Time>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx);

    template<typename FormatContext>
    auto format(Time const& time, FormatContext& ctx);
};

template<typename ParseContext>
constexpr auto fmt::formatter<Time>::parse(ParseContext& ctx)
{
    return ctx.begin();
}


template<>
struct fmt::formatter<Date>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx);

    template<typename FormatContext>
    auto format(Date const& date, FormatContext& ctx);
};

template<typename ParseContext>
constexpr auto fmt::formatter<Date>::parse(ParseContext& ctx)
{
    return ctx.begin();
}


class Formatter
{
    static std::unique_ptr<Formatter> m_instance;

    Language m_language;
    std::locale m_locale;

    std::vector<std::string> m_weekDayNames;
    std::vector<std::string> m_monthNames;

    explicit Formatter(Language language);

    DECLARE_NOT_COPYABLE(Formatter);
    DECLARE_NOT_MOVABLE(Formatter);

public:
    static void initialize(Language language);
    static void clear();  // For tests only

    template<class... Types>
    static std::string format(const std::string& f, Types... args);

    static Language language();

    static const std::vector<std::string>& weekDayNames();
    static const std::vector<std::string>& monthNames();

protected:
    template<typename FormatContext>
    static auto format(Time const& time, FormatContext& ctx);

    template<typename FormatContext>
    static auto format(Date const& date, FormatContext& ctx);

    friend fmt::formatter<Time>;
    friend fmt::formatter<Date>;
};

template<class... Types>
std::string Formatter::format(const std::string& f, Types... args)
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is are not initialized.");
    }

    return fmt::vformat(m_instance->m_locale, f, fmt::make_format_args(args...));
}

inline Language Formatter::language()
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is not initialized.");
    }

    return m_instance->m_language;
}

inline const std::vector<std::string>& Formatter::weekDayNames()
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is not initialized.");
    }

    return m_instance->m_weekDayNames;
}

inline const std::vector<std::string>& Formatter::monthNames()
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is not initialized.");
    }

    return m_instance->m_monthNames;
}

template<typename FormatContext>
auto fmt::formatter<Time>::format(Time const& time, FormatContext& ctx)
{
    return Formatter::format(time, ctx);
}

template<typename FormatContext>
auto Formatter::format(Time const& time, FormatContext& ctx)
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is not initialized.");
    }

    if (m_instance->m_language == Language::ENGLISH)
    {
        int hour = time.hour > 12 ? time.hour - 12 : time.hour;
        return fmt::format_to(ctx.out(), "{}:{:02d} {}", hour, time.minute, time.hour < 12 ? "AM" : "PM");
    }
    else if (m_instance->m_language == Language::FRENCH)
    {
        return fmt::format_to(ctx.out(), "{}:{:02d}", time.hour, time.minute);
    }
    else
    {
        throw std::runtime_error("Invalid language");
    }
}

template<typename FormatContext>
auto fmt::formatter<Date>::format(Date const& date, FormatContext& ctx)
{
    return Formatter::format(date, ctx);
}

template<typename FormatContext>
auto Formatter::format(Date const& date, FormatContext& ctx)
{
    if (m_instance == nullptr)
    {
        throw std::runtime_error("The formatter is not initialized.");
    }

    int month = std::max(0, std::min(date.month, 11));

    if (m_instance->m_language == Language::ENGLISH)
    {
        return fmt::format_to(ctx.out(), "{} {}, {}", m_instance->m_monthNames[month], date.day, date.year);
    }
    else if (m_instance->m_language == Language::FRENCH)
    {
        return fmt::format_to(ctx.out(), "{} {} {}", date.day, m_instance->m_monthNames[month], date.year);
    }
    else
    {
        throw std::runtime_error("Invalid language");
    }
}

#endif
