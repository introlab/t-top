#ifndef SMART_SPEAKER_STATES_WAIT_PERSON_IDENTIFICATION_STATE_H
#define SMART_SPEAKER_STATES_WAIT_PERSON_IDENTIFICATION_STATE_H

#include "State.h"

#include <person_identification/PersonNames.h>

class WaitPersonIdentificationState : public State
{
    ros::Subscriber m_personNamesSubscriber;
    ros::Timer m_timeoutTimer;

public:
    WaitPersonIdentificationState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~WaitPersonIdentificationState() override = default;

    DECLARE_NOT_COPYABLE(WaitPersonIdentificationState);
    DECLARE_NOT_MOVABLE(WaitPersonIdentificationState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index WaitPersonIdentificationState::type() const
{
    return std::type_index(typeid(WaitPersonIdentificationState));
}

#endif
