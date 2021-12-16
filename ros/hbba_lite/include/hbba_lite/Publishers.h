#ifndef HBBA_LITE_PUBLISHERS_H
#define HBBA_LITE_PUBLISHERS_H

#include <ros/ros.h>

#include <hbba_lite/FilterState.h>

template <class FilterState, class MessageType>
class HbbaPublisher
{
    FilterState m_filterState;
    ros::Publisher m_publisher;

public:
    HbbaPublisher(ros::NodeHandle& nodeHandle,
            const std::string& topic, uint32_t queueSize,
            const std::string& stateServiceName = "", bool latch = false);

    uint32_t getNumSubscribers() const;
    std::string getTopic() const;
    bool isLatched() const;
    void shutdown();

    bool isFilteringAllMessages() const;

    void publish(const MessageType& msg);
};

template <class FilterState, class MessageType>
inline HbbaPublisher<FilterState, MessageType>::HbbaPublisher(ros::NodeHandle& nodeHandle,
            const std::string& topic, uint32_t queueSize,
            const std::string& stateServiceName, bool latch) :
    m_filterState(nodeHandle, stateServiceName == "" ? topic + "/filter_state" : stateServiceName)
{
    m_publisher = nodeHandle.advertise<MessageType>(topic, queueSize, latch);
}

template <class FilterState, class MessageType>
inline uint32_t HbbaPublisher<FilterState, MessageType>::getNumSubscribers() const
{
    return m_publisher.getNumSubscribers();
}

template <class FilterState, class MessageType>
inline bool HbbaPublisher<FilterState, MessageType>::isLatched() const
{
    return m_publisher.isLatched();
}

template <class FilterState, class MessageType>
std::string HbbaPublisher<FilterState, MessageType>::getTopic() const
{
    return m_publisher.getTopic();
}

template <class FilterState, class MessageType>
inline void HbbaPublisher<FilterState, MessageType>::shutdown()
{
    m_publisher.shutdown();
}

template <class FilterState, class MessageType>
bool HbbaPublisher<FilterState, MessageType>::isFilteringAllMessages() const
{
    return m_filterState.isFilteringAllMessages();
}

template <class FilterState, class MessageType>
inline void HbbaPublisher<FilterState, MessageType>::publish(const MessageType& msg)
{
    if (m_filterState.check())
    {
        m_publisher.publish(msg);
    }
}

template <class MessageType>
using OnOffHbbaPublisher = HbbaPublisher<OnOffHbbaFilterState, MessageType>;

template <class MessageType>
using ThrottlingHbbaPublisher = HbbaPublisher<ThrottlingHbbaFilterState, MessageType>;

#endif
