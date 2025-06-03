#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>


#include <behavior_srvs/srv/chat_tools_function_call.hpp>
#include <daemon_ros_client/msg/base_status.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <cloud_data/srv/current_local_weather_open_meteo.hpp>         // Weather service definition
#include <cloud_data/srv/local_weather_forecast_open_meteo.hpp>   
#include <perceptions_analyzer/srv/perceive_objects.hpp>   



#include <memory>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <ctime>

using json = nlohmann::json;
using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "chatbot_node";

void publish_volume(uint8_t volume, rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr volumePublisher)
{
    std_msgs::msg::UInt8 msg;
    msg.data = volume;
    volumePublisher->publish(msg);
}

int startNode()
{
    auto node = rclcpp::Node::make_shared(NODE_NAME);
    auto callbackGroup = node->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    daemon_ros_client::msg::BaseStatus::SharedPtr baseStatusMsg;
    auto volumePublisher = node->create_publisher<std_msgs::msg::UInt8>("daemon/set_volume", 1);

    // Create service for chat tools function call
    auto service_volume_up = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_up",
        [node, &baseStatusMsg, volumePublisher](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(rclcpp::get_logger(NODE_NAME), "Received service volume_up request");
            // Handle the service request here
            try
            {
                json j = json::parse(request->function_arguments);
                uint8_t amount = j["amount"];

                if (baseStatusMsg)
                {
                    // Test volume higher limit
                    if (amount + baseStatusMsg->volume > baseStatusMsg->maximum_volume)
                    {
                        amount = baseStatusMsg->maximum_volume - baseStatusMsg->volume;
                        RCLCPP_WARN(
                            rclcpp::get_logger(NODE_NAME),
                            fmt::format(
                                "Volume cannot be higher than {0}. Will increase by {1} instead.",
                                baseStatusMsg->maximum_volume,
                                amount)
                                .c_str());
                    }

                    uint8_t volume = amount + baseStatusMsg->volume;
                    response->ok = true;
                    response->result = fmt::format(
                        "{{\"status\": \"Volume increased by {0} to {1} over {2}\"}}",
                        amount,
                        volume,
                        baseStatusMsg->maximum_volume);
                    publish_volume(volume, volumePublisher);
                }
                else
                {
                    response->ok = false;
                    response->result = fmt::format(
                        "{{\"status\": \"Could not increase volume, current volume: {0}\"}}",
                        baseStatusMsg->volume);
                }
            }
            catch (const json::parse_error& e)
            {
                RCLCPP_ERROR(node->get_logger(), "JSON parse error: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    auto service_volume_down = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_down",
        [node, &baseStatusMsg, volumePublisher](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(rclcpp::get_logger(NODE_NAME), "Received service volume_down request");
            // Handle the service request here
            try
            {
                json j = json::parse(request->function_arguments);
                uint8_t amount = j["amount"];
                if (baseStatusMsg)
                {
                    // Test volume lower limit
                    if (amount > baseStatusMsg->volume)
                    {
                        amount = baseStatusMsg->volume;
                        RCLCPP_WARN(
                            rclcpp::get_logger(NODE_NAME),
                            fmt::format("Volume cannot be lower than 0. Will decrease by {0} instead.", amount).c_str());
                    }
                    uint8_t volume = baseStatusMsg->volume - amount;
                    response->ok = true;
                    response->result = fmt::format(
                        "{{\"status\": \"Volume decreased by {0} to {1} over {2}\"}}",
                        amount,
                        volume,
                        baseStatusMsg->maximum_volume);
                    publish_volume(volume, volumePublisher);
                }
                else
                {
                    response->ok = false;
                    response->result = fmt::format(
                        "{{\"status\": \"Could not decrease volume, current volume: {0} \"}}",
                        baseStatusMsg->volume);
                }
            }
            catch (const json::parse_error& e)
            {
                RCLCPP_ERROR(node->get_logger(), "JSON parse error: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    rclcpp::Client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>::SharedPtr weather_client =
    node->create_client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>("/cloud_data/current_local_weather");

    auto service_get_weather = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_current_weather",
        [node, weather_client](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(node->get_logger(), "Received get_current_weather request");
    
            // Check if weather service is available
            if (!weather_client->wait_for_service(std::chrono::seconds(2))) {
                RCLCPP_ERROR(node->get_logger(), "Weather service unavailable");
                response->ok = false;
                response->result = "{\"error\": \"Weather service unavailable\"}";
                return;
            }
    
            try {
                auto req = std::make_shared<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Request>();
                
                // Use a promise/future pattern instead of spin_until_future_complete
                std::promise<std::shared_ptr<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Response>> promise;
                std::future<std::shared_ptr<cloud_data::srv::CurrentLocalWeatherOpenMeteo::Response>> future = promise.get_future();
                
                auto callback = [&promise](rclcpp::Client<cloud_data::srv::CurrentLocalWeatherOpenMeteo>::SharedFuture inner_future) {
                    promise.set_value(inner_future.get());
                };
                
                // Send the request with a callback
                weather_client->async_send_request(req, callback);
                
                // Wait for the response with a timeout
                auto status = future.wait_for(std::chrono::seconds(5));
                
                if (status != std::future_status::ready) {
                    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for weather response");
                    response->ok = false;
                    response->result = "{\"error\": \"Timeout\"}";
                    return;
                }
                
                auto res = future.get();
                if (res && res->ok) {
                    // Using nlohmann::json
                    nlohmann::json payload = {
                        {"city", res->city},
                        {"region", res->region},
                        {"country", res->country_name},
                        {"temperature_celsius", res->temperature_celsius},
                        {"wind_speed_kph", res->wind_speed_kph}
                    };
                    response->ok = true;
                    response->result = payload.dump();
                    RCLCPP_INFO(node->get_logger(), "Weather data sent successfully");
                } else {
                    RCLCPP_ERROR(node->get_logger(), "Weather service returned error");
                    response->ok = false;
                    response->result = "{\"error\": \"Failed to get weather\"}";
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(node->get_logger(), "Exception in weather service callback: %s", e.what());
                response->ok = false;
                response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
            } catch (...) {
                RCLCPP_ERROR(node->get_logger(), "Unknown exception in weather service callback");
                response->ok = false;
                response->result = "{\"error\": \"Unknown internal error\"}";
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    rclcpp::Client<cloud_data::srv::LocalWeatherForecastOpenMeteo>::SharedPtr forecast_client =
    node->create_client<cloud_data::srv::LocalWeatherForecastOpenMeteo>("/cloud_data/local_weather_forecast");
    
    auto service_get_forecast = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_local_forecast",
        [node, forecast_client](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(node->get_logger(), "Received get_local_forecast request");
    
            // Default to today (relative_day = 0)
            uint64_t relative_day = 0;
            
            // Try to access the arguments field - modify this part based on your actual request structure
            try {
                // Log the request structure to help debug
                RCLCPP_INFO(node->get_logger(), "Examining request structure");
                
                // Check if there's a 'function_arguments' field or similar
                // This is a placeholder - you need to find the correct field name
                if (request->function_arguments.length() > 0) {
                    nlohmann::json args = nlohmann::json::parse(request->function_arguments);
                    if (args.contains("relative_day")) {
                        relative_day = args["relative_day"].get<uint64_t>();
                    }
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(node->get_logger(), "Failed to parse arguments: %s", e.what());
                // Continue with default value
            }
            
            RCLCPP_INFO(node->get_logger(), "Processing forecast for relative day: %ld", relative_day);
    
            // Check if forecast service is available
            if (!forecast_client->wait_for_service(std::chrono::seconds(2))) {
                RCLCPP_ERROR(node->get_logger(), "Forecast service unavailable");
                response->ok = false;
                response->result = "{\"error\": \"Forecast service unavailable\"}";
                return;
            }
    
            try {
                auto req = std::make_shared<cloud_data::srv::LocalWeatherForecastOpenMeteo::Request>();
                req->relative_day = relative_day;
                
                // Use a promise/future pattern
                std::promise<std::shared_ptr<cloud_data::srv::LocalWeatherForecastOpenMeteo::Response>> promise;
                std::future<std::shared_ptr<cloud_data::srv::LocalWeatherForecastOpenMeteo::Response>> future = promise.get_future();
                
                auto callback = [&promise](rclcpp::Client<cloud_data::srv::LocalWeatherForecastOpenMeteo>::SharedFuture inner_future) {
                    promise.set_value(inner_future.get());
                };
                
                // Send the request with a callback
                forecast_client->async_send_request(req, callback);
                
                // Wait for the response with a timeout
                auto status = future.wait_for(std::chrono::seconds(5));
                
                if (status != std::future_status::ready) {
                    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for forecast response");
                    response->ok = false;
                    response->result = "{\"error\": \"Timeout\"}";
                    return;
                }
                
                auto res = future.get();
                if (res && res->ok) {
                    // Create a descriptive forecast day name
                    std::string day_description;
                    if (relative_day == 0) {
                        day_description = "Today";
                    } else if (relative_day == 1) {
                        day_description = "Tomorrow";
                    } else {
                        day_description = "Day " + std::to_string(relative_day);
                    }
                    
                    // Using nlohmann::json
                    nlohmann::json payload = {
                        {"day", day_description},
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
                        {"sunset", res->sunset}
                    };
                    response->ok = true;
                    response->result = payload.dump();
                    RCLCPP_INFO(node->get_logger(), "Forecast data sent successfully for day %ld", relative_day);
                } else {
                    RCLCPP_ERROR(node->get_logger(), "Forecast service returned error");
                    response->ok = false;
                    response->result = "{\"error\": \"Failed to get forecast\"}";
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(node->get_logger(), "Exception in forecast service callback: %s", e.what());
                response->ok = false;
                response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
            } catch (...) {
                RCLCPP_ERROR(node->get_logger(), "Unknown exception in forecast service callback");
                response->ok = false;
                response->result = "{\"error\": \"Unknown internal error\"}";
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);
    

    rclcpp::Client<perceptions_analyzer::srv::PerceiveObjects>::SharedPtr perception_client =
    node->create_client<perceptions_analyzer::srv::PerceiveObjects>("/perception/detected_objects");

    auto service_get_perceive_objects = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_perceive_objects",
        [node, perception_client](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(node->get_logger(), "Received get_perceive_objects request");
    
            // Check if weather service is available
            if (!perception_client->wait_for_service(std::chrono::seconds(2))) {
                RCLCPP_ERROR(node->get_logger(), "Perceive objects service unavailable");
                response->ok = false;
                response->result = "{\"error\": \"Perceive objects service unavailable\"}";
                return;
            }
    
            try {
                auto req = std::make_shared<perceptions_analyzer::srv::PerceiveObjects::Request>();
                
                // Use a promise/future pattern instead of spin_until_future_complete
                std::promise<std::shared_ptr<perceptions_analyzer::srv::PerceiveObjects::Response>> promise;
                std::future<std::shared_ptr<perceptions_analyzer::srv::PerceiveObjects::Response>> future = promise.get_future();
                
                auto callback = [&promise](rclcpp::Client<perceptions_analyzer::srv::PerceiveObjects>::SharedFuture inner_future) {
                    promise.set_value(inner_future.get());
                };
                
                // Send the request with a callback
                perception_client->async_send_request(req, callback);
                
                // Wait for the response with a timeout
                auto status = future.wait_for(std::chrono::seconds(5));
                
                if (status != std::future_status::ready) {
                    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for erceive objects response");
                    response->ok = false;
                    response->result = "{\"error\": \"Timeout\"}";
                    return;
                }
                
                auto res = future.get();
                if (res && res->ok) {
                    // Using nlohmann::json
                    nlohmann::json payload = {
                        {"objects", res->objects}
                    };
                    response->ok = true;
                    response->result = payload.dump();
                    RCLCPP_INFO(node->get_logger(), "Perceive objects data sent successfully");
                } else {
                    RCLCPP_ERROR(node->get_logger(), "Perceive objects service returned error");
                    response->ok = false;
                    response->result = "{\"error\": \"Failed to get weather\"}";
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(node->get_logger(), "Exception in perceive objects service callback: %s", e.what());
                response->ok = false;
                response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
            } catch (...) {
                RCLCPP_ERROR(node->get_logger(), "Unknown exception in perceive objects service callback");
                response->ok = false;
                response->result = "{\"error\": \"Unknown internal error\"}";
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    auto service_get_date = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_date_and_time",
        [node ](
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request>,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
        {
            RCLCPP_INFO(rclcpp::get_logger(NODE_NAME), "Received service get_date_and_time request");
            try
            {
                time_t timestamp = std::time(nullptr);
                nlohmann::json payload = {
                    {"date and time", std::ctime(&timestamp)},
                };
                response->ok = true;
                response->result = payload.dump();
                RCLCPP_INFO(node->get_logger(), "Date data sent successfully");

                
            }
            catch (const json::parse_error& e)
            {
                RCLCPP_ERROR(node->get_logger(), "JSON parse error: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Invalid JSON format: {0}\"}}", e.what());
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
                response->ok = false;
                response->result = fmt::format("{{\"status\": \"Exception: {0}\"}}", e.what());
            }
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    

    rclcpp::SubscriptionOptions options;
    options.callback_group = callbackGroup;
    auto baseStatusSubscriber = node->create_subscription<daemon_ros_client::msg::BaseStatus>(
        "daemon/base_status",
        1,
        [node, &baseStatusMsg](const daemon_ros_client::msg::BaseStatus::SharedPtr msg) { baseStatusMsg = msg; },
        options);

    auto desireSet = make_shared<DesireSet>();
    auto rosFilterPool = make_unique<RosFilterPool>(node, WAIT_FOR_SERVICE);
    auto filterPool = make_shared<RosLogFilterPoolDecorator>(node, move(rosFilterPool));

    vector<unique_ptr<BaseStrategy>> strategies;

    strategies.emplace_back(createChatStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createTooCloseReactionStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dWithAnalyzedImageStrategy(filterPool));



    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    desireSet->addDesire(make_unique<ChatDesire>());
    desireSet->addDesire(make_unique<NearestFaceFollowingDesire>());
    //desireSet->addDesire(make_unique<TooCloseReactionDesire>());
    desireSet->addDesire(make_unique<FastVideoAnalyzer3dWithAnalyzedImageDesire>());


    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);

    RCLCPP_INFO_STREAM(node->get_logger(), "Chatbot started");
    executor.add_node(node);
    executor.spin();
    return 0;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        return startNode();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger(NODE_NAME), "Chatbot crashed (" << e.what() << ")");
        return -1;
    }

    rclcpp::shutdown();
}
