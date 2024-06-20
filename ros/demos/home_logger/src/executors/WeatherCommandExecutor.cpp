#include "WeatherCommandExecutor.h"

#include "../states/common/TalkState.h"
#include "../states/specific/IdleState.h"

#include <home_logger_common/DateTime.h>
#include <home_logger_common/language/Formatter.h>
#include <home_logger_common/language/StringResources.h>

#include <sstream>

using namespace std;

constexpr chrono::seconds WEATHER_SERVICE_TIMEOUT(20);

WeatherCommandExecutor::WeatherCommandExecutor(StateManager& stateManager, rclcpp::Node::SharedPtr node)
    : SpecificCommandExecutor<WeatherCommand>(stateManager),
      m_node(move(node))
{
    m_weatherClientCallbackGroup = m_node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    m_currentWeatherClient = m_node->create_client<cloud_data::srv::CurrentLocalWeather>(
        "cloud_data/current_local_weather",
        rmw_qos_profile_services_default,
        m_weatherClientCallbackGroup);
    m_weatherForecastClient = m_node->create_client<cloud_data::srv::LocalWeatherForecast>(
        "cloud_data/local_weather_forecast",
        rmw_qos_profile_services_default,
        m_weatherClientCallbackGroup);
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
    auto request = make_shared<cloud_data::srv::CurrentLocalWeather::Request>();
    auto future = m_currentWeatherClient->async_send_request(request);

    if (future.wait_for(WEATHER_SERVICE_TIMEOUT) != future_status::ready || !future.get()->ok)
    {
        ok = false;
        return;
    }

    ok = true;
    text = Formatter::format(
        StringResources::getValue("dialogs.commands.weather.current"),
        fmt::arg("temperature_celsius", future.get()->temperature_celsius),
        fmt::arg("weather_description", future.get()->weather_description));
}

void WeatherCommandExecutor::getTodayWeatherForecastText(string& text, bool& ok)
{
    auto request = make_shared<cloud_data::srv::LocalWeatherForecast::Request>();
    request->relative_day = 0;
    auto future = m_weatherForecastClient->async_send_request(request);

    if (future.wait_for(WEATHER_SERVICE_TIMEOUT) != future_status::ready || !future.get()->ok)
    {
        ok = false;
        return;
    }

    ok = true;

    stringstream ss;
    auto currentTime = Time::now();
    if (currentTime < Time(12, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.morning"),
            fmt::arg("temperature_celsius", future.get()->temperature_morning_celsius));
        ss << "\n";
    }
    if (currentTime < Time(17, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.day"),
            fmt::arg("temperature_celsius", future.get()->temperature_day_celsius));
        ss << "\n";
    }
    if (currentTime < Time(21, 00))
    {
        ss << Formatter::format(
            StringResources::getValue("dialogs.commands.weather.today.evening"),
            fmt::arg("temperature_celsius", future.get()->temperature_evening_celsius));
        ss << "\n";
    }

    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.today.night"),
        fmt::arg("temperature_celsius", future.get()->temperature_night_celsius));

    text = ss.str();
}

void WeatherCommandExecutor::getTomorrowWeatherForecastText(string& text, bool& ok)
{
    auto request = make_shared<cloud_data::srv::LocalWeatherForecast::Request>();
    request->relative_day = 1;
    auto future = m_weatherForecastClient->async_send_request(request);

    if (future.wait_for(WEATHER_SERVICE_TIMEOUT) != future_status::ready || !future.get()->ok)
    {
        ok = false;
        return;
    }

    ok = true;

    stringstream ss;
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.morning"),
        fmt::arg("temperature_celsius", future.get()->temperature_morning_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.day"),
        fmt::arg("temperature_celsius", future.get()->temperature_day_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.evening"),
        fmt::arg("temperature_celsius", future.get()->temperature_evening_celsius));
    ss << "\n";
    ss << Formatter::format(
        StringResources::getValue("dialogs.commands.weather.tomorrow.night"),
        fmt::arg("temperature_celsius", future.get()->temperature_night_celsius));

    text = ss.str();
}

void WeatherCommandExecutor::getWeekWeatherForecastText(string& text, bool& ok)
{
    constexpr int DAY_COUNT = 7;
    float temperatures[DAY_COUNT];

    auto request = make_shared<cloud_data::srv::LocalWeatherForecast::Request>();

    for (size_t i = 0; i < DAY_COUNT; i++)
    {
        request->relative_day = i;
        auto future = m_weatherForecastClient->async_send_request(request);
        if (future.wait_for(WEATHER_SERVICE_TIMEOUT) != future_status::ready || !future.get()->ok)
        {
            ok = false;
            return;
        }

        temperatures[i] = future.get()->temperature_day_celsius;
    }

    ok = true;

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
