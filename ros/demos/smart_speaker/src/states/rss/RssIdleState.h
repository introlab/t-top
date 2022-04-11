#ifndef SMART_SPEAKER_STATES_RSS_RSS_IDLE_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_IDLE_STATE_H

#include "../State.h"

#include <std_msgs/Empty.h>
#include <person_identification/PersonNames.h>

class RssIdleState : public State
{
    ros::Subscriber m_robotNameDetectedSubscriber;
    ros::Subscriber m_personNamesSubscriber;

public:
    RssIdleState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssIdleState() override = default;

    DECLARE_NOT_COPYABLE(RssIdleState);
    DECLARE_NOT_MOVABLE(RssIdleState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;

private:
    void robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg);
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
};

inline std::type_index RssIdleState::type() const
{
    return std::type_index(typeid(RssIdleState));
}

#endif
