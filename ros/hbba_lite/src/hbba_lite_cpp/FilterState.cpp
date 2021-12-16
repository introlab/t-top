#include <hbba_lite/FilterState.h>

using namespace std;

OnOffHbbaFilterState::OnOffHbbaFilterState(ros::NodeHandle& nodeHandle, const std::string& stateServiceName) :
    m_isFilteringAllMessages(true)
{
    m_stateService = nodeHandle.advertiseService(stateServiceName, &OnOffHbbaFilterState::stateServiceCallback, this);
}

bool OnOffHbbaFilterState::check()
{
    return !m_isFilteringAllMessages;
}

bool OnOffHbbaFilterState::stateServiceCallback(hbba_lite::SetOnOffFilterState::Request &request,
        hbba_lite::SetOnOffFilterState::Response &response)
{
    m_isFilteringAllMessages = request.is_filtering_all_messages;
    response.ok = true;
    return true;
}

ThrottlingHbbaFilterState::ThrottlingHbbaFilterState(ros::NodeHandle& nodeHandle, const std::string& stateServiceName) :
    m_isFilteringAllMessages(true), m_rate(1), m_counter(0)
{
    m_stateService = nodeHandle.advertiseService(stateServiceName, &ThrottlingHbbaFilterState::stateServiceCallback, this);
}

bool ThrottlingHbbaFilterState::check()
{
    if (m_isFilteringAllMessages)
    {
        return false;
    }

    bool isReady = false;
    if (m_counter == 0)
    {
        isReady = true;
    }
    m_counter = (m_counter + 1) % m_rate;
    return isReady;
}

bool ThrottlingHbbaFilterState::stateServiceCallback(hbba_lite::SetThrottlingFilterState::Request &request,
        hbba_lite::SetThrottlingFilterState::Response &response)
{
    if (request.rate <= 0)
    {
        response.ok = false;
    }
    else
    {
        m_isFilteringAllMessages = request.is_filtering_all_messages;
        m_rate = request.rate;
        m_counter = 0;

        response.ok = true;
    }

    return true;
}
