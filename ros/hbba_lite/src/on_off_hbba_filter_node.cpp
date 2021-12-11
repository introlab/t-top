#include <ros/ros.h>
#include <hbba_lite/HbbaFilterNode.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "on_off_hbba_filter_node");
    ros::NodeHandle nodeHandle;

    OnOffHbbaFilterNode onOffHbbaFilterNode(nodeHandle);
    onOffHbbaFilterNode.run();

    return 0;
}
