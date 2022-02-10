#ifndef SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H

#include "../State.h"

#include <std_msgs/Empty.h>
#include <person_identification/PersonNames.h>
#include <video_analyzer/VideoAnalysis.h>

class SmartIdleState : public State
{
    double m_personDistanceThreshold;

    ros::Subscriber m_personNamesSubscriber;
    ros::Subscriber m_videoAnalysisSubscriber;

public:
    SmartIdleState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        double personDistanceThreshold);
    ~SmartIdleState() override = default;

    DECLARE_NOT_COPYABLE(SmartIdleState);
    DECLARE_NOT_MOVABLE(SmartIdleState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;

private:
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
    void videoAnalysisSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg);
};

inline std::type_index SmartIdleState::type() const
{
    return std::type_index(typeid(SmartIdleState));
}

#endif
