
#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>


#include <behavior_srvs/srv/chat_tools_function_call.hpp>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "chatbot_node";

int startNode() {

    auto node = rclcpp::Node::make_shared(NODE_NAME);
    auto callbackGroup = node->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // Create service for chat tools function call
    auto service_volume_up = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_up",
        [](const std::shared_ptr<rmw_request_id_t> request_header,
           const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
           const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response) {
            RCLCPP_INFO(rclcpp::get_logger(NODE_NAME), "Received service volume_up request");
            // Handle the service request here
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    auto service_volume_down = node->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/volume_down",
        [](const std::shared_ptr<rmw_request_id_t> request_header,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
            const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response) {
            RCLCPP_INFO(rclcpp::get_logger(NODE_NAME), "Received service volume_down request");
            // Handle the service request here
        },
        rmw_qos_profile_services_default,
        callbackGroup);

    auto desireSet = make_shared<DesireSet>();

    auto rosFilterPool = make_unique<RosFilterPool>(node, WAIT_FOR_SERVICE);
    auto filterPool = make_shared<RosLogFilterPoolDecorator>(node, move(rosFilterPool));

    vector<unique_ptr<BaseStrategy>> strategies;

    //strategies.emplace_back(createSpeechToTextStrategy(filterPool));
    //strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
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
