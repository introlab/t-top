#include "ControlPanel.h"

#include <QApplication>

#include <ros/ros.h>

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "control_panel_node");
    ros::AsyncSpinner spinner(1);

    ros::NodeHandle nodeHandle;

    QApplication a(argc, argv);
    ControlPanel controlPanel(nodeHandle);
    controlPanel.show();

    spinner.start();
    int returnCode = a.exec();
    spinner.stop();

    return returnCode;
}
