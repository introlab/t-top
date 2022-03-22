#ifndef HBBA_LITE_FILTERS_SUBSCRIBER_H
#define HBBA_LITE_FILTERS_SUBSCRIBER_H

#include <ros/ros.h>

#include <hbba_lite/filters/FilterState.h>

template<class FilterState, class MessageType>
class HbbaSubscriber
{
    FilterState m_filterState;
    boost::function<void(const typename MessageType::ConstPtr&)> m_callback;
    ros::Subscriber m_subscriber;

public:
    template<class T>
    HbbaSubscriber(
        ros::NodeHandle& nodeHandle,
        const std::string& topic,
        uint32_t queueSize,
        void (T::*fp)(const typename MessageType::ConstPtr&),
        T* obj,
        const std::string& stateServiceName = "",
        const ros::TransportHints& transportHints = ros::TransportHints());
    HbbaSubscriber(
        ros::NodeHandle& nodeHandle,
        const std::string& topic,
        uint32_t queueSize,
        void (*fp)(const typename MessageType::ConstPtr&),
        const std::string& stateServiceName = "",
        const ros::TransportHints& transportHints = ros::TransportHints());

    uint32_t getNumPublishers() const;
    std::string getTopic() const;
    void shutdown();

    bool isFilteringAllMessages() const;

private:
    void callback(const typename MessageType::ConstPtr& msg);
};

template<class FilterState, class MessageType>
template<class T>
HbbaSubscriber<FilterState, MessageType>::HbbaSubscriber(
    ros::NodeHandle& nodeHandle,
    const std::string& topic,
    uint32_t queueSize,
    void (T::*fp)(const typename MessageType::ConstPtr&),
    T* obj,
    const std::string& stateServiceName,
    const ros::TransportHints& transportHints)
    : m_filterState(nodeHandle, stateServiceName == "" ? topic + "/filter_state" : stateServiceName)
{
    m_callback = boost::bind(fp, obj, _1);
    m_subscriber =
        nodeHandle
            .subscribe(topic, queueSize, &HbbaSubscriber<FilterState, MessageType>::callback, this, transportHints);
}

template<class FilterState, class MessageType>
HbbaSubscriber<FilterState, MessageType>::HbbaSubscriber(
    ros::NodeHandle& nodeHandle,
    const std::string& topic,
    uint32_t queueSize,
    void (*fp)(const typename MessageType::ConstPtr&),
    const std::string& stateServiceName,
    const ros::TransportHints& transportHints)
    : m_filterState(nodeHandle, stateServiceName == "" ? topic + "/filter_state" : stateServiceName)
{
    m_callback = fp;
    m_subscriber =
        nodeHandle
            .subscribe(topic, queueSize, &HbbaSubscriber<FilterState, MessageType>::callback, this, transportHints);
}

template<class FilterState, class MessageType>
uint32_t HbbaSubscriber<FilterState, MessageType>::getNumPublishers() const
{
    return m_subscriber.getNumPublishers();
}

template<class FilterState, class MessageType>
std::string HbbaSubscriber<FilterState, MessageType>::getTopic() const
{
    return m_subscriber.getTopic();
}

template<class FilterState, class MessageType>
void HbbaSubscriber<FilterState, MessageType>::shutdown()
{
    m_subscriber.shutdown();
}

template<class FilterState, class MessageType>
bool HbbaSubscriber<FilterState, MessageType>::isFilteringAllMessages() const
{
    return m_filterState.isFilteringAllMessages();
}

template<class FilterState, class MessageType>
void HbbaSubscriber<FilterState, MessageType>::callback(const typename MessageType::ConstPtr& msg)
{
    if (m_filterState.check() && m_callback)
    {
        m_callback(msg);
    }
}

template<class MessageType>
using OnOffHbbaSubscriber = HbbaSubscriber<OnOffHbbaFilterState, MessageType>;

template<class MessageType>
using ThrottlingHbbaSubscriber = HbbaSubscriber<ThrottlingHbbaFilterState, MessageType>;

#endif
