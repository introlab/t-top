#include "widgets/ControlPanel.h"

#include <QApplication>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/RosStrategyStateLogger.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

int startNode(int argc, char* argv[])
{
    ros::init(argc, argv, "control_panel_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    bool camera2dWideEnabled = false;
    privateNodeHandle.param("camera_2d_wide_enabled", camera2dWideEnabled, false);

    auto desireSet = make_shared<DesireSet>();
    auto rosFilterPool = make_unique<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);
    auto filterPool = make_shared<RosLogFilterPoolDecorator>(move(rosFilterPool));

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dWithAnalyzedImageStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createVadStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createLedEmotionStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createLedAnimationStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSpecificFaceFollowingStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createSoundObjectPersonFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createTooCloseReactionStrategy(filterPool));

    if (camera2dWideEnabled)
    {
        strategies.emplace_back(createFastVideoAnalyzer2dWideWithAnalyzedImageStrategy(filterPool));
    }

    auto solver = make_unique<GecodeSolver>();
    auto strategyStateLogger = make_unique<RosLogStrategyStateLogger>();
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver), move(strategyStateLogger));

    QApplication application(argc, argv);
    ControlPanel controlPanel(nodeHandle, desireSet, camera2dWideEnabled);
    // controlPanel.setFixedSize(600, 900);
    controlPanel.show();

    ros::AsyncSpinner spinner(1);
    spinner.start();
    int returnCode = application.exec();
    spinner.stop();

    return returnCode;
}

int main(int argc, char* argv[])
{
    try
    {
        return startNode(argc, argv);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Control panel crashed (" << e.what() << ")");
        return -1;
    }
}
