#include <ros/ros.h>
#include <std_msgs/Int8.h>

#include <hbba_lite/Subscribers.h>

void callback(const std_msgs::Int8::ConstPtr& msg)
{
  ROS_INFO("Data received : %i", static_cast<int>(msg->data));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_on_off_hbba_subscriber");
    ros::NodeHandle nodeHandle;

    ThrottlingHbbaSubscriber<std_msgs::Int8> sub(nodeHandle, "int_topic", 10, &callback);
    ros::spin();

    return 0;
}
