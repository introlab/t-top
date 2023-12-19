#include "Connect4Widget.h"

#include <QApplication>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top_hbba_lite/Desires.h>
#include <t_top_hbba_lite/Strategies.h>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

int startNode(int argc, char* argv[])
{
    ros::init(argc, argv, "control_panel_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    bool fullscreen = false;
    privateNodeHandle.param("fullscreen", fullscreen, false);

    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createTelepresenceStrategy(filterPool));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createLedAnimationStrategy(filterPool, desireSet, nodeHandle));

    strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));


    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"sound", 1}}, move(solver));

    QApplication application(argc, argv);
    Connect4Widget connect4Widget(nodeHandle, desireSet);
    if (fullscreen)
    {
        connect4Widget.setWindowState(Qt::WindowFullScreen);
    }
    else
    {
        connect4Widget.setMinimumSize(300, 300);
    }
    connect4Widget.show();

    ros::AsyncSpinner spinner(1);
    spinner.start();

    desireSet->addDesire<TelepresenceDesire>();

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
