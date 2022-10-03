#include "widgets/ControlPanel.h"

#include <QApplication>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
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
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dWithAnalyzedImageStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSpecificFaceFollowingStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createSoundObjectPersonFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createDanceStrategy(filterPool));

    if (camera2dWideEnabled)
    {
        strategies.emplace_back(createFastVideoAnalyzer2dWideWithAnalyzedImageStrategy(filterPool));
    }

    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"motor", 1}, {"sound", 1}}, move(solver));

    QApplication application(argc, argv);
    ControlPanel controlPanel(nodeHandle, desireSet, camera2dWideEnabled);
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
        ROS_ERROR_STREAM("Smart speaker crashed (" << e.what() << ")");
        return -1;
    }
}
