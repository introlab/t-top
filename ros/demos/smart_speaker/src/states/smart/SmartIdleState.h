#ifndef SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H

#include "../State.h"

#include <tf/transform_listener.h>

#include <std_msgs/Empty.h>
#include <person_identification/PersonNames.h>
#include <video_analyzer/VideoAnalysis.h>

class SmartIdleState : public State
{
    double m_personDistanceThreshold;
    std::string m_personDistanceFrameId;
    double m_noseConfidenceThreshold;
    size_t m_videoAnalysisMessageCountThreshold;
    size_t m_videoAnalysisMessageCountTolerance;

    size_t m_videoAnalysisValidMessageCount;
    size_t m_videoAnalysisInvalidMessageCount;

    tf::TransformListener m_tfListener;

    ros::Subscriber m_personNamesSubscriber;
    ros::Subscriber m_videoAnalysisSubscriber;

public:
    SmartIdleState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        double personDistanceThreshold,
        std::string personDistanceFrameId,
        double noseConfidenceThreshold,
        size_t videoAnalysisMessageCountThreshold,
        size_t videoAnalysisMessageCountTolerance);
    ~SmartIdleState() override = default;

    DECLARE_NOT_COPYABLE(SmartIdleState);
    DECLARE_NOT_MOVABLE(SmartIdleState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;

private:
    void personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg);
    void videoAnalysisSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg);

    double personNameDistance(const person_identification::PersonName& name);
    double faceDistance(const video_analyzer::VideoAnalysisObject& object, const tf::StampedTransform& transform);
};

inline std::type_index SmartIdleState::type() const
{
    return std::type_index(typeid(SmartIdleState));
}

#endif
