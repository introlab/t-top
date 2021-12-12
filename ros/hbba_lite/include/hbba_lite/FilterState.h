#ifndef HBBA_LITE_FILTER_STATE
#define HBBA_LITE_FILTER_STATE

#include <ros/ros.h>
#include <hbba_lite/SetOnOffFilterState.h>
#include <hbba_lite/SetThrottlingFilterState.h>

#include <string>

class OnOffHbbaFilterState
{
    ros::ServiceServer m_stateService;
    bool m_isFilteringAllMessages;

public:
    OnOffHbbaFilterState(ros::NodeHandle& nodeHandle, const std::string& stateServiceName);
    bool check();
    bool isFilteringAllMessages() const;

private:
    bool stateServiceCallback(hbba_lite::SetOnOffFilterState::Request &request,
            hbba_lite::SetOnOffFilterState::Response &response);
};

bool OnOffHbbaFilterState::isFilteringAllMessages() const
{
    return m_isFilteringAllMessages;
}

class ThrottlingHbbaFilterState
{
    ros::ServiceServer m_stateService;
    bool m_isFilteringAllMessages;
    int m_rate;
    int m_counter;

public:
    ThrottlingHbbaFilterState(ros::NodeHandle& nodeHandle, const std::string& stateServiceName);
    bool check();
    bool isFilteringAllMessages() const;

private:
    bool stateServiceCallback(hbba_lite::SetThrottlingFilterState::Request &request,
            hbba_lite::SetThrottlingFilterState::Response &response);
};

bool ThrottlingHbbaFilterState::isFilteringAllMessages() const
{
    return m_isFilteringAllMessages;
}

#endif
