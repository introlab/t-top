#ifndef SMART_SPEAKER_STATES_IDLE_STATE_H
#define SMART_SPEAKER_STATES_IDLE_STATE_H

#include "State.h"

#include <std_msgs/Empty.h>
#include <person_identification/PersonNames.h>

class IdleState : public State
{
    ros::Subscriber m_robotNameDetectedSubscriber;
    ros::Subscriber m_personNamesSubscriber;

public:
    IdleState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~IdleState() override = default;

    DECLARE_NOT_COPYABLE(IdleState);
    DECLARE_NOT_MOVABLE(IdleState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;

private:
    void robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg);
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
};

inline std::type_index IdleState::type() const
{
    return std::type_index(typeid(IdleState));
}

#endif
