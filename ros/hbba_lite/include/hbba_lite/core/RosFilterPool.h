#ifndef HBBA_LITE_CORE_ROS_STRATEGY_H
#define HBBA_LITE_CORE_ROS_STRATEGY_H

#include <hbba_lite/core/Strategy.h>

#include <ros/ros.h>

#include <unordered_map>

class RosFilterPool : public FilterPool
{
    ros::NodeHandle& m_nodeHandle;
    bool m_waitForService;

    std::unordered_map<std::string, ros::ServiceClient> m_serviceClientsByName;

public:
    RosFilterPool(ros::NodeHandle& nodeHandle, bool waitForService);
    void add(const std::string& name, FilterType type) override;

protected:
    void applyEnabling(const std::string& name, const FilterConfiguration& configuration) override;
    void applyDisabling(const std::string& name) override;

private:
    template<class ServiceType>
    void call(const std::string& name, ServiceType& request);
};

template<class ServiceType>
void RosFilterPool::call(const std::string& name, ServiceType& srv)
{
    ros::ServiceClient& service = m_serviceClientsByName[name];
    if (m_waitForService)
    {
        ros::service::waitForService(name);
    }

    if (!service.isValid())
    {
        service = m_nodeHandle.serviceClient<ServiceType>(name, true);
    }
    if (!service.exists())
    {
        ROS_ERROR("The service does not exist (%s)", name.c_str());
    }
    else if (!service.call(srv) || !srv.response.ok)
    {
        ROS_ERROR("The service call has failed (%s)", name.c_str());
    }
}


#endif
