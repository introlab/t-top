#include "WeatherCommandExecutor.h"

#include "../states/common/TalkState.h"
#include "../states/specific/IdleState.h"

#include <home_logger_common/DateTime.h>
#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include <cloud_data/CurrentLocalWeather.h>
#include <cloud_data/LocalWeatherForecast.h>

#include <sstream>

using namespace std;

WeatherCommandExecutor::WeatherCommandExecutor(StateManager& stateManager, ros::NodeHandle& nodeHandle)
    : SpecificCommandExecutor<WeatherCommand>(stateManager),
      m_nodeHandle(nodeHandle)
{
}

WeatherCommandExecutor::~WeatherCommandExecutor() {}

void WeatherCommandExecutor::executeSpecific(const shared_ptr<WeatherCommand>& command)
{
    string text;
    bool ok = false;

    switch (command->time().value())
    {
        case WeatherTime::CURRENT:
            getCurrentWeatherText(text, ok);
            break;

        case WeatherTime::TODAY:
            getTodayWeatherForecastText(text, ok);
            break;

        case WeatherTime::TOMORROW:
            getTomorrowWeatherForecastText(text, ok);
            break;

        case WeatherTime::WEEK:
            getWeekWeatherForecastText(text, ok);
            break;

        default:
            throw runtime_error("Invalid weather time");
    }

    if (ok)
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            text,
            "",  // No gesture
            "blink",
            StateType::get<TalkState>(),
            getAskNextCommandParameter()));
    }
    else
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.commands.weather.not_available"),
            "",  // No gesture
            "sad",
            StateType::get<TalkState>(),
            getAskNextCommandParameter()));
    }
}

void WeatherCommandExecutor::getCurrentWeatherText(string& text, bool& ok)
{
    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::CurrentLocalWeather>("cloud_data/current_local_weather");

    cloud_data::CurrentLocalWeather srv;
    ok = service.exists() && service.call(srv);
    if (!ok)
    {
        return;
    }

    text = Formatter::format(
        StringResources::getValue("dialogs.commands.weather.current"),
        fmt::arg("temperature_celsius", srv.response.temperature_celsius),
        fmt::arg("weather_description", srv.response.weather_description));
}

void WeatherCommandExecutor::getTodayWeatherForecastText(string& text, bool& ok)
{
    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::LocalWeatherForecast>("cloud_data/local_weather_forecast");

    cloud_data::LocalWeatherForecast srv;
    srv.request.relative_day = 0;
    ok = service.exists() && service.call(srv);
    if (!ok)
    {
        return;
    }

    stringstream ss;
    auto currentTime = Time::now();
    if (currentTime < Time(12, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.morning"),
            fmt::arg("temperature_celsius", srv.response.temperature_morning_celsius));
        ss << "\n";
    }
    if (currentTime < Time(17, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.day"),
            fmt::arg("temperature_celsius", srv.response.temperature_day_celsius));
        ss << "\n";
    }
    if (currentTime < Time(21, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.evening"),
            fmt::arg("temperature_celsius", srv.response.temperature_evening_celsius));
        ss << "\n";
    }

    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.today.night"),
        fmt::arg("temperature_celsius", srv.response.temperature_night_celsius));

    text = ss.str();
}

void WeatherCommandExecutor::getTomorrowWeatherForecastText(string& text, bool& ok)
{
    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::LocalWeatherForecast>("cloud_data/local_weather_forecast");

    cloud_data::LocalWeatherForecast srv;
    srv.request.relative_day = 0;
    ok = service.exists() && service.call(srv);
    if (!ok)
    {
        return;
    }

    stringstream ss;
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.morning"),
        fmt::arg("temperature_celsius", srv.response.temperature_morning_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.day"),
        fmt::arg("temperature_celsius", srv.response.temperature_day_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.evening"),
        fmt::arg("temperature_celsius", srv.response.temperature_evening_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.night"),
        fmt::arg("temperature_celsius", srv.response.temperature_night_celsius));

    text = ss.str();
}

void WeatherCommandExecutor::getWeekWeatherForecastText(string& text, bool& ok)
{
    constexpr int DAY_COUNT = 7;
    float temperatures[DAY_COUNT];

    ros::ServiceClient service =
        m_nodeHandle.serviceClient<cloud_data::LocalWeatherForecast>("cloud_data/local_weather_forecast");
    for (size_t i = 0; i < DAY_COUNT; i++)
    {
        cloud_data::LocalWeatherForecast srv;
        srv.request.relative_day = i;
        ok = service.exists() && service.call(srv);
        if (!ok)
        {
            return;
        }

        temperatures[i] = srv.response.temperature_day_celsius;
    }

    int currentWeekDay = Date::now().weekDay();
    string temperatureSentence = StringResources::getValue("dialogs.commands.weather.week.temperature");

    stringstream ss;
    for (int i = 0; i <= DAY_COUNT; i++)
    {
        if (i == 0)
        {
            if (Time::now() >= Time(12, 00))
            {
                continue;
            }
            ss << StringResources::getValue("dialogs.commands.weather.week.today");
        }
        else if (i == 1)
        {
            ss << StringResources::getValue("dialogs.commands.weather.week.tomorrow");
        }
        else
        {
            ss << Formatter::weekDayNames().at((currentWeekDay + i) % DAY_COUNT);
        }

        ss << ", ";
        ss << Formatter::format(temperatureSentence, fmt::arg("temperature_celsius", temperatures[i]));
        ss << "\n";
    }

    text = ss.str();
}
