#include "Connect4Widget.h"

#include <QApplication>

#include <rclcpp/rclcpp.hpp>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top_hbba_lite/Desires.h>
#include <t_top_hbba_lite/Strategies.h>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "connnect4_node";

int startNode(int argc, char* argv[])
{
    auto node = rclcpp::Node::make_shared(NODE_NAME);

    bool fullscreen = node->declare_parameter("fullscreen", false);

    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(node, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createTelepresenceStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createLedAnimationStrategy(filterPool, desireSet, node));

    strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));


    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver));

    QApplication application(argc, argv);
    Connect4Widget connect4Widget(node, desireSet);
    if (fullscreen)
    {
        connect4Widget.setWindowState(Qt::WindowFullScreen);
    }
    else
    {
        connect4Widget.setMinimumSize(300, 300);
    }
    connect4Widget.show();

    // Run ROS in background
    rclcpp::executors::SingleThreadedExecutor rosExecutor;
    rosExecutor.add_node(node);
    std::thread spinThread([&rosExecutor]() { rosExecutor.spin(); });

    desireSet->addDesire<TelepresenceDesire>();

    int returnCode = application.exec();
    rosExecutor.cancel();
    spinThread.join();

    return returnCode;
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    try
    {
        return startNode(argc, argv);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger(NODE_NAME), "Control panel crashed (" << e.what() << ")");
        return -1;
    }

    rclcpp::shutdown();
}
