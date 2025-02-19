
#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "chatbot_node";

int startNode() {

    auto node = rclcpp::Node::make_shared(NODE_NAME);
    auto desireSet = make_shared<DesireSet>();

    auto rosFilterPool = make_unique<RosFilterPool>(node, WAIT_FOR_SERVICE);
    auto filterPool = make_shared<RosLogFilterPoolDecorator>(node, move(rosFilterPool));

    vector<unique_ptr<BaseStrategy>> strategies;

    //strategies.emplace_back(createSpeechToTextStrategy(filterPool));
    //strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createChatStrategy(filterPool, desireSet, node));


    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosTopicStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));


    desireSet->addDesire(make_unique<ChatDesire>());

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
