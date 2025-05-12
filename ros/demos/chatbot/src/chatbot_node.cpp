#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>


#include <behavior_srvs/srv/chat_tools_function_call.hpp>
#include <daemon_ros_client/msg/base_status.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <memory>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>

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


    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    desireSet->addDesire(make_unique<ChatDesire>());
    desireSet->addDesire(make_unique<NearestFaceFollowingDesire>());
    desireSet->addDesire(make_unique<TooCloseReactionDesire>());

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
