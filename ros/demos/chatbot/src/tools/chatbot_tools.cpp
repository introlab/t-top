#include <chatbot/tools/chatbot_tools.hpp>

#include <memory>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <ctime>


using json = nlohmann::json;
using namespace std;


ChatbotTools::ChatbotTools(shared_ptr<rclcpp::Node> node, rclcpp::CallbackGroup::SharedPtr callbackGroup)
    : node_(std::move(node)),
      callback_group_(callbackGroup)
{
    volume_publisher_ = node_->create_publisher<std_msgs::msg::UInt8>("daemon/set_volume", 1);

    rclcpp::SubscriptionOptions options;
    options.callback_group = callbackGroup;

    base_status_subscriber_ = node_->create_subscription<daemon_ros_client::msg::BaseStatus>(
        "daemon/base_status",
        1,
        std::bind(&ChatbotTools::on_base_status_, this, std::placeholders::_1),
        options);

    service_volume_up_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_up",
        std::bind(&ChatbotTools::handle_volume_up_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    service_volume_down_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_down",
        std::bind(&ChatbotTools::handle_volume_down_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    service_get_weather_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_current_weather",
        std::bind(&ChatbotTools::handle_get_weather_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    service_get_forecast_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_local_forecast",
        std::bind(&ChatbotTools::handle_get_forecast_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    service_get_perceived_objects_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_perceived_objects",
        std::bind(&ChatbotTools::handle_perceive_objects_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    service_get_date_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_date_and_time",
        std::bind(&ChatbotTools::handle_get_date_request, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    weather_client_ = node_->create_client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>(
        "/cloud_data/open_meteo/current_local_weather");
    forecast_client_ = node_->create_client<cloud_data::srv::LocalWeatherForecastOpenMeteo>(
        "/cloud_data/open_meteo/local_weather_forecast");
    perceive_objects_client_ =
        node_->create_client<perception_msgs::srv::PerceiveObjects>("/perception/detected_objects");
}

void ChatbotTools::on_base_status_(const daemon_ros_client::msg::BaseStatus::SharedPtr msg)
{
    base_status_msg_ = msg;
}

void ChatbotTools::publish_volume(uint8_t volume)
{
    std_msgs::msg::UInt8 msg;
    msg.data = volume;
    volume_publisher_->publish(msg);
}

void ChatbotTools::handle_volume_up_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received service volume_up request");
    // Handle the service request here
    try
    {
        json j = json::parse(request->function_arguments);
        uint8_t amount = j["amount"];

        if (base_status_msg_)
        {
            // Test volume higher limit
            if (amount + base_status_msg_->volume > base_status_msg_->maximum_volume)
            {
                amount = base_status_msg_->maximum_volume - base_status_msg_->volume;
                RCLCPP_WARN(
                    node_->get_logger(),
                    fmt::format(
                        "Volume cannot be higher than {0}. Will increase by {1} instead.",
                        base_status_msg_->maximum_volume,
                        amount)
                        .c_str());
            }

            uint8_t volume = amount + base_status_msg_->volume;
            response->ok = true;
            response->result = fmt::format(
                "{{\"status\": \"Volume increased by {0} to {1} over {2}\"}}",
                amount,
                volume,
                base_status_msg_->maximum_volume);
            publish_volume(volume);
        }
        else
        {
            response->ok = false;
            response->result = fmt::format(
                "{{\"status\": \"Could not increase volume, current volume: {0}\"}}",
                base_status_msg_->volume);
        }
    }
    catch (const json::parse_error& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "JSON parse error: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
    }
}

void ChatbotTools::handle_volume_down_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received service volume_down request");
    // Handle the service request here
    try
    {
        json j = json::parse(request->function_arguments);
        uint8_t amount = j["amount"];
        if (base_status_msg_)
        {
            // Test volume lower limit
            if (amount > base_status_msg_->volume)
            {
                amount = base_status_msg_->volume;
                RCLCPP_WARN(
                    node_->get_logger(),
                    fmt::format("Volume cannot be lower than 0. Will decrease by {0} instead.", amount).c_str());
            }
            uint8_t volume = base_status_msg_->volume - amount;
            response->ok = true;
            response->result = fmt::format(
                "{{\"status\": \"Volume decreased by {0} to {1} over {2}\"}}",
                amount,
                volume,
                base_status_msg_->maximum_volume);
            publish_volume(volume);
        }
        else
        {
            response->ok = false;
            response->result = fmt::format(
                "{{\"status\": \"Could not decrease volume, current volume: {0} \"}}",
                base_status_msg_->volume);
        }
    }
    catch (const json::parse_error& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "JSON parse error: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
    }
}

void ChatbotTools::handle_get_weather_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received get_current_weather request");

    // Check if weather service is available
    if (!weather_client_->wait_for_service(std::chrono::seconds(2)))
    {
        RCLCPP_ERROR(node_->get_logger(), "Weather service unavailable");
        response->ok = false;
        response->result = "{\"error\": \"Weather service unavailable\"}";
        return;
    }

    try
    {
        auto req = std::make_shared<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Request>();

        std::promise<std::shared_ptr<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Response>> promise;
        std::future<std::shared_ptr<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Response>> future =
            promise.get_future();

        auto callback =
            [&promise](rclcpp::Client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>::SharedFuture inner_future)
        { promise.set_value(inner_future.get()); };

        weather_client_->async_send_request(req, callback);

        auto status = future.wait_for(std::chrono::seconds(5));

        if (status != std::future_status::ready)
        {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for weather response");
            response->ok = false;
            response->result = "{\"error\": \"Timeout\"}";
            return;
        }

        auto res = future.get();
        if (res && res->ok)
        {
            // Using nlohmann::json
            nlohmann::json payload = {
                {"city", res->city},
                {"region", res->region},
                {"country", res->country_name},
                {"temperature_celsius", res->temperature_celsius},
                {"feels_like_temperature_celsius", res->feels_like_temperature_celsius},
                {"wind_speed_kph", res->wind_speed_kph},
                {"clouds_cover", res->clouds_cover},
                {"precipitation", res->precipitation},
                {"snowfall", res->snowfall},
                {"precipitation_probability_percent", res->precipitation_probability_percent},
                {"rain", res->rain},
            };
            response->ok = true;
            response->result = payload.dump();
            RCLCPP_INFO(node_->get_logger(), "Weather data sent successfully");
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "Weather service returned error");
            response->ok = false;
            response->result = "{\"error\": \"Failed to get weather\"}";
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception in weather service callback: %s", e.what());
        response->ok = false;
        response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
    }
}

void ChatbotTools::handle_get_forecast_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received get_local_forecast request");

    uint64_t relative_day = 0;
    try
    {
        RCLCPP_INFO(node_->get_logger(), "Examining request structure");

        if (request->function_arguments.length() > 0)
        {
            nlohmann::json args = nlohmann::json::parse(request->function_arguments);
            if (args.contains("relative_day"))
            {
                relative_day = args["relative_day"].get<uint64_t>();
            }
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_WARN(node_->get_logger(), "Failed to parse arguments: %s", e.what());
    }

    RCLCPP_INFO(node_->get_logger(), "Processing forecast for relative day: %ld", relative_day);

    if (!forecast_client_->wait_for_service(std::chrono::seconds(10)))
    {
        RCLCPP_ERROR(node_->get_logger(), "Forecast service unavailable");
        response->ok = false;
        response->result = "{\"error\": \"Forecast service unavailable\"}";
        return;
    }

    try
    {
        auto req = std::make_shared<cloud_data::srv::LocalWeatherForecastOpenMeteo::Request>();
        req->relative_day = relative_day;

        std::promise<std::shared_ptr<cloud_data::srv::LocalWeatherForecastOpenMeteo::Response>> promise;
        std::future<std::shared_ptr<cloud_data::srv::LocalWeatherForecastOpenMeteo::Response>> future =
            promise.get_future();

        auto callback =
            [&promise](rclcpp::Client<cloud_data::srv::LocalWeatherForecastOpenMeteo>::SharedFuture inner_future)
        { promise.set_value(inner_future.get()); };

        forecast_client_->async_send_request(req, callback);

        auto status = future.wait_for(std::chrono::seconds(5));

        if (status != std::future_status::ready)
        {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for forecast response");
            response->ok = false;
            response->result = "{\"error\": \"Timeout\"}";
            return;
        }

        auto res = future.get();
        if (res && res->ok)
        {
            nlohmann::json payload = {
                {"city", res->city},
                {"region", res->region},
                {"country", res->country_name},
                {"temperature_day_celsius", res->temperature_day_celsius},
                {"temperature_night_celsius", res->temperature_night_celsius},
                {"feels_like_temperature_day_celsius", res->feals_like_temperature_day_celsius},
                {"feels_like_temperature_night_celsius", res->feals_like_temperature_night_celsius},
                {"precipitation_sum", res->precipitation_sum},
                {"precipitation_probability_percent", res->precipitation_probability_percent},
                {"wind_speed_kph", res->wind_speed_kph},
                {"sunrise", res->sunrise},
                {"sunset", res->sunset}};
            response->ok = true;
            response->result = payload.dump();
            RCLCPP_INFO(node_->get_logger(), "Forecast data sent successfully for day %ld", relative_day);
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "Forecast service returned error");
            response->ok = false;
            response->result = "{\"error\": \"Failed to get forecast\"}";
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception in forecast service callback: %s", e.what());
        response->ok = false;
        response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
    }
}

void ChatbotTools::handle_perceive_objects_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received get_perceived_objects request");

    // Check if weather service is available
    if (!perceive_objects_client_->wait_for_service(std::chrono::seconds(2)))
    {
        RCLCPP_ERROR(node_->get_logger(), "Perceive objects service unavailable");
        response->ok = false;
        response->result = "{\"error\": \"Perceive objects service unavailable\"}";
        return;
    }

    try
    {
        auto req = std::make_shared<perception_msgs::srv::PerceiveObjects::Request>();

        // Use a promise/future pattern instead of spin_until_future_complete
        std::promise<std::shared_ptr<perception_msgs::srv::PerceiveObjects::Response>> promise;
        std::future<std::shared_ptr<perception_msgs::srv::PerceiveObjects::Response>> future = promise.get_future();

        auto callback = [&promise](rclcpp::Client<perception_msgs::srv::PerceiveObjects>::SharedFuture inner_future)
        { promise.set_value(inner_future.get()); };

        perceive_objects_client_->async_send_request(req, callback);

        auto status = future.wait_for(std::chrono::seconds(5));

        if (status != std::future_status::ready)
        {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for rerceive objects response");
            response->ok = false;
            response->result = "{\"error\": \"Timeout\"}";
            return;
        }

        auto res = future.get();
        if (res && res->ok)
        {
            nlohmann::json payload = {{"objects", res->objects}};
            response->ok = true;
            response->result = payload.dump();
            RCLCPP_INFO(node_->get_logger(), "Perceive objects data sent successfully");
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "Perceive objects service returned error");
            response->ok = false;
            response->result = "{\"error\": \"Failed to get weather\"}";
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception in perceive objects service callback: %s", e.what());
        response->ok = false;
        response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
    }
}

void ChatbotTools::handle_get_date_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received service get_date_and_time request");
    try
    {
        time_t timestamp = std::time(nullptr);
        char timeString[21];
        std::strftime(timeString, sizeof(timeString), "%Y-%m-%dT%H:%M:%SZ", std::localtime(&timestamp));
        nlohmann::json payload = {"date and time", timeString};
        response->ok = true;
        response->result = payload.dump();
        RCLCPP_INFO(node_->get_logger(), "Date data sent successfully");
    }
    catch (const json::parse_error& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "JSON parse error: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception: %s", e.what());
        response->ok = false;
        response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
    }
}
