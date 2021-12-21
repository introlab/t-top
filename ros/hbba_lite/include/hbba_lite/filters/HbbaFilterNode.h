#ifndef HBBA_FILTER_FILTERS_NODE_H
#define HBBA_FILTER_FILTERS_NODE_H

#include <ros/ros.h>
#include <topic_tools/shape_shifter.h>

#include <hbba_lite/filters/FilterState.h>

// Inspired by https://github.com/ros/ros_comm/blob/indigo-devel/tools/topic_tools/src/relay.cpp
template <class FilterState>
class HbbaFilterNode
{
    ros::NodeHandle& m_nodeHandle;
    FilterState m_filterState;
    ros::Subscriber m_subscriber;
    ros::Publisher m_publisher;
    bool m_hasAdvertised;

public:
    HbbaFilterNode(ros::NodeHandle& nodeHandle);
    void run();

private:
    void callback(const ros::MessageEvent<topic_tools::ShapeShifter>& msg_event);
};

template <class FilterState>
inline HbbaFilterNode<FilterState>::HbbaFilterNode(ros::NodeHandle& nodeHandle) :
    m_nodeHandle(nodeHandle),
    m_filterState(nodeHandle, "filter_state"),
    m_hasAdvertised(false)
{
    m_subscriber = nodeHandle.subscribe("in", 10, &HbbaFilterNode<FilterState>::callback, this);
}

template <class FilterState>
inline void HbbaFilterNode<FilterState>::run()
{
    ros::spin();
}

template <class FilterState>
void HbbaFilterNode<FilterState>::callback(const ros::MessageEvent<topic_tools::ShapeShifter>& msg_event)
{
    boost::shared_ptr<topic_tools::ShapeShifter const> const &msg = msg_event.getConstMessage();

    if (!m_hasAdvertised)
    {
        boost::shared_ptr<const ros::M_string> const& connectionHeader = msg_event.getConnectionHeaderPtr();
        bool latch = false;
        if (connectionHeader)
        {
            auto it = connectionHeader->find("latching");
            if(it != connectionHeader->end() && it->second == "1")
            {
                latch = true;
            }
        }

        m_publisher = msg->advertise(m_nodeHandle, "out", 10, latch);
        m_hasAdvertised = true;
    }

    if (m_filterState.check())
    {
        m_publisher.publish(msg);
    }
}

typedef HbbaFilterNode<OnOffHbbaFilterState> OnOffHbbaFilterNode;
typedef HbbaFilterNode<ThrottlingHbbaFilterState> ThrottlingHbbaFilterNode;

#endif
