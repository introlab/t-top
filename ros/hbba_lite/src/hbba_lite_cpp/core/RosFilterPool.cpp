#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/utils/HbbaLiteException.h>

#include <hbba_lite/SetOnOffFilterState.h>
#include <hbba_lite/SetThrottlingFilterState.h>

using namespace std;

RosFilterPool::RosFilterPool(ros::NodeHandle& nodeHandle) : m_nodeHandle(nodeHandle)
{
}

void RosFilterPool::add(const string& name, FilterType type)
{
    lock_guard<recursive_mutex> lock(m_mutex);
    FilterPool::add(name, type);

    switch (type)
    {
    case FilterType::ON_OFF:
        m_serviceClientsByName[name] = m_nodeHandle.serviceClient<hbba_lite::SetOnOffFilterState>(name, true);
        break;

    case FilterType::THROTTLING:
        m_serviceClientsByName[name] = m_nodeHandle.serviceClient<hbba_lite::SetThrottlingFilterState>(name, true);
        break;

    default:
        throw HbbaLiteException("Not supported filter type");
    }
}

void RosFilterPool::applyEnabling(const string& name, const FilterConfiguration& configuration)
{
    switch (m_typesByName[name])
    {
    case FilterType::ON_OFF:
        {
            hbba_lite::SetOnOffFilterState srv;
            srv.request.is_filtering_all_messages = false;
            call(name, srv);
        }
        break;

    case FilterType::THROTTLING:
        {
            hbba_lite::SetThrottlingFilterState srv;
            srv.request.is_filtering_all_messages = false;
            srv.request.rate = configuration.rate();
            call(name, srv);
        }
        break;

    default:
        throw HbbaLiteException("Not supported filter type");
    }
}

void RosFilterPool::applyDisabling(const string& name)
{
    switch (m_typesByName[name])
    {
    case FilterType::ON_OFF:
        {
            hbba_lite::SetOnOffFilterState srv;
            srv.request.is_filtering_all_messages = true;
            call(name, srv);
        }
        break;

    case FilterType::THROTTLING:
        {
            hbba_lite::SetThrottlingFilterState srv;
            srv.request.is_filtering_all_messages = true;
            srv.request.rate = 1;
            call(name, srv);
        }
        break;

    default:
        throw HbbaLiteException("Not supported filter type");
    }
}
