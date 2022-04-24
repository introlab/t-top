#ifndef SMART_SPEAKER_STATES_RSS_RSS_WAIT_PERSON_IDENTIFICATION_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_WAIT_PERSON_IDENTIFICATION_STATE_H

#include "../State.h"

#include <person_identification/PersonNames.h>

class RssWaitPersonIdentificationState : public State
{
    ros::Subscriber m_personNamesSubscriber;
    ros::Timer m_timeoutTimer;

public:
    RssWaitPersonIdentificationState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssWaitPersonIdentificationState() override = default;

    DECLARE_NOT_COPYABLE(RssWaitPersonIdentificationState);
    DECLARE_NOT_MOVABLE(RssWaitPersonIdentificationState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

private:
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
    void timeoutTimerCallback(const ros::TimerEvent& event);
};

inline std::type_index RssWaitPersonIdentificationState::type() const
{
    return std::type_index(typeid(RssWaitPersonIdentificationState));
}

#endif
