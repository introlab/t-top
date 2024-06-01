#include "widgets/ControlPanel.h"

#include <QApplication>

#include <rclcpp/rclcpp.hpp>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;
constexpr const char* NODE_NAME = "control_panel_node";

int startNode(int argc, char* argv[])
{
    auto node = rclcpp::Node::make_shared(NODE_NAME);

    bool camera2dWideEnabled = node->declare_parameter("camera_2d_wide_enabled", false);

    auto desireSet = make_shared<DesireSet>();
    auto rosFilterPool = make_unique<RosFilterPool>(node, WAIT_FOR_SERVICE);
    auto filterPool = make_shared<RosLogFilterPoolDecorator>(node, move(rosFilterPool));

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dWithAnalyzedImageStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createVadStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, node));
    strategies.emplace_back(createLedEmotionStrategy(filterPool, node));
    strategies.emplace_back(createLedAnimationStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSpecificFaceFollowingStrategy(filterPool, node));
    strategies.emplace_back(createSoundObjectPersonFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, node));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createTooCloseReactionStrategy(filterPool));

    if (camera2dWideEnabled)
    {
        strategies.emplace_back(createFastVideoAnalyzer2dWideWithAnalyzedImageStrategy(filterPool));
    }

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosLogStrategyStateLogger>(node);
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    QApplication application(argc, argv);
    ControlPanel controlPanel(node, desireSet, camera2dWideEnabled);
    // controlPanel.setFixedSize(600, 900);
    controlPanel.show();

    // Run ROS in background
    rclcpp::executors::SingleThreadedExecutor rosExecutor;
    rosExecutor.add_node(node);
    std::thread spinThread([&rosExecutor]() { rosExecutor.spin(); });

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
