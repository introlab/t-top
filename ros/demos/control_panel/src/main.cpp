#include "widgets/ControlPanel.h"

#include <QApplication>

#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>

#include <memory>

using namespace std;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "control_panel_node");
    ros::NodeHandle nodeHandle;

    auto desireSet = make_shared<DesireSet>();

    QApplication a(argc, argv);
    ControlPanel controlPanel(nodeHandle, desireSet);
    controlPanel.show();

    ros::AsyncSpinner spinner(1);
    spinner.start();
    int returnCode = a.exec();
    spinner.stop();

    return returnCode;
}
