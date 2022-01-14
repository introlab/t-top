#include "widgets/ControlPanel.h"
#include "ControlPanelStrategies.h"

#include <QApplication>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <memory>

using namespace std;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "control_panel_node");
    ros::NodeHandle nodeHandle;

    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle);

    vector<unique_ptr<BaseStrategy>> strategies;
    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createTalkStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createListenStrategy(filterPool));
    strategies.emplace_back(createGestureStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createDanceStrategy(filterPool));
    strategies.emplace_back(createExploreStrategy(filterPool));
    strategies.emplace_back(createVideoAnalyzerStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));

    auto solver = make_unique<GecodeSolver>();
    HbbaLite hbba(desireSet, move(strategies), {{"motor", 1}, {"sound", 1}}, move(solver));

    QApplication a(argc, argv);
    ControlPanel controlPanel(nodeHandle, desireSet);
    controlPanel.show();

    ros::AsyncSpinner spinner(1);
    spinner.start();
    int returnCode = a.exec();
    spinner.stop();

    return returnCode;
}
