#include <ros/ros.h>
#include <hbba_lite/HbbaFilterNode.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "throttling_hbba_filter_node");
    ros::NodeHandle nodeHandle;

    ThrottlingHbbaFilterNode throttlingHbbaFilterNode(nodeHandle);
    throttlingHbbaFilterNode.run();

    return 0;
}
