#ifndef HOME_LOGGER_EXECUTORS_WEATHER_COMMAND_EXECUTORS_H
#define HOME_LOGGER_EXECUTORS_WEATHER_COMMAND_EXECUTORS_H

#include "CommandExecutor.h"

class WeatherCommandExecutor : public SpecificCommandExecutor<WeatherCommand>
{
    rclcpp::Node::SharedPtr m_node;

public:
    WeatherCommandExecutor(StateManager& stateManager, rclcpp::Node::SharedPtr node);
    ~WeatherCommandExecutor() override;

protected:
    void executeSpecific(const std::shared_ptr<WeatherCommand>& command) override;

private:
    void getCurrentWeatherText(std::string& text, bool& ok);
    void getTodayWeatherForecastText(std::string& text, bool& ok);
    void getTomorrowWeatherForecastText(std::string& text, bool& ok);
    void getWeekWeatherForecastText(std::string& text, bool& ok);
};

#endif
