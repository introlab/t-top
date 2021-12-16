#include <ros/ros.h>
#include <std_msgs/Int8.h>

#include <hbba_lite/Publishers.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_on_off_hbba_publisher");
    ros::NodeHandle nodeHandle;

    OnOffHbbaPublisher<std_msgs::Int8> pub(nodeHandle, "int_topic", 10);

    ros::Rate rate(1);
    for (int i = 0; ros::ok(); i++)
    {
        std_msgs::Int8 msg;
        msg.data = i;
        pub.publish(msg);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
