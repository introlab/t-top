# hbba_lite

This is a lite version of [HBBA](https://github.com/francoisferland/hbba).

The main differences are :

- Desires cannot have dependencies;
- All Motivations modules are inside the same ROS node;
- The filters can be integrated in ROS nodes;
- Two kinds of filters exist: on/off and throttling;
- It use [Gecode](https://www.gecode.org/) instead of [OR-Tools](https://developers.google.com/optimization) for the
  solver.

## How to use?

### Create Motivation Modules

See the [demos](../demos).

### Adding a Filter to C++ Nodes

#### Subscriber

```cpp
#include <ros/ros.h>
#include <std_msgs/Int8.h>

#include <hbba_lite/filters/Subscribers.h>

void callback(const std_msgs::Int8::ConstPtr& msg)
{
    ROS_INFO("Data received : %i", static_cast<int>(msg->data));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "node_name");
    ros::NodeHandle nodeHandle;

    // Replace this
    ros::Subscriber sub = n.subscribe("int_topic", 10, callback);

    // with this to add an on/off filter
    OnOffHbbaSubscriber<std_msgs::Int8> sub(nodeHandle, "int_topic", 10, &callback);

    // with this to add a throttling filter
    ThrottlingHbbaSubscriber<std_msgs::Int8> sub(nodeHandle, "int_topic", 10, &callback);
    ros::spin();

    return 0;
}
```

#### Publisher

```cpp
#include <ros/ros.h>
#include <std_msgs/Int8.h>

#include <hbba_lite/filters/Publishers.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "node_name");
    ros::NodeHandle nodeHandle;

    // Replace this
    ros::Publisher pub = n.advertise<std_msgs::Int8>("int_topic", 10);

    // with this to add an on/off filter
    OnOffHbbaPublisher<std_msgs::Int8> pub(nodeHandle, "int_topic", 10);

    // with this to add a throttling filter
    ThrottlingHbbaPublisher<std_msgs::Int8> pub(nodeHandle, "int_topic", 10);

    return 0;
}

```

### Adding a Filter to Python Nodes

#### Subscriber

```python
import rospy
from std_msgs.msg import Int8
import hbba_lite


def callback(data):
    rospy.loginfo('Data received : {}'.format(data.data))


def main():
    rospy.init_node('node_name')

    # Replace this
    sub = rospy.Subscriber("int_topic", Int8, callback)

    # with this to add an on/off filter
    sub = hbba_lite.OnOffHbbaSubscriber('int_topic', Int8, callback)

    # with this to add a throttling filter
    sub = hbba_lite.ThrottlingHbbaSubscriber('int_topic', Int8, callback)

    rospy.spin()


if __name__ == '__main__':
    main()

```

#### Publisher

```python
import rospy
from std_msgs.msg import Int8
import hbba_lite


def main():
    rospy.init_node('node_name')

    # Replace this
    pub = rospy.Publisher('int_topic', Int8, queue_size=10)

    # with this to add an on/off filter
    pub = hbba_lite.OnOffHbbaPublisher('int_topic', Int8, queue_size=10)

    # with this to add a throttling filter
    pub = hbba_lite.ThrottlingHbbaPublisher('int_topic', Int8, queue_size=10)


if __name__ == '__main__':
    main()

```

## Nodes

### `on_off_hbba_filter_node`

This node applies an on/off filter on a topic.

#### Subscribed Topics

- `in` (Any): The input topic.
- `out` (Any): The filtered topic.

#### Services

- `filter_state` ([hbba_lite/SetOnOffFilterState](srv/SetOnOffFilterState.srv)) The service to change the filter state.

### `throttling_hbba_filter_node`

This node applies a throttling filter on a topic.

#### Subscribed Topics

- `in` (Any): The input topic.
- `out` (Any): The filtered topic.

#### Services

- `filter_state` ([hbba_lite/SetThrottlingFilterState](srv/SetThrottlingFilterState.srv)) The service to change the
  filter state.
