#ifndef SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H
#define SMART_SPEAKER_STATES_SMART_SMART_IDLE_STATE_H

#include "../State.h"

#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <person_identification/msg/person_names.hpp>
#include <video_analyzer/msg/video_analysis.hpp>

class SmartIdleState : public State
{
    double m_personDistanceThreshold;
    std::string m_personDistanceFrameId;
    double m_noseConfidenceThreshold;
    size_t m_videoAnalysisMessageCountThreshold;
    size_t m_videoAnalysisMessageCountTolerance;

    size_t m_videoAnalysisValidMessageCount;
    size_t m_videoAnalysisInvalidMessageCount;

    std::unique_ptr<tf2_ros::Buffer> m_tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

    rclcpp::Subscription<person_identification::msg::PersonNames>::SharedPtr m_personNamesSubscriber;
    rclcpp::Subscription<video_analyzer::msg::VideoAnalysis>::SharedPtr m_videoAnalysisSubscriber;

public:
    SmartIdleState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node,
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
    void personNamesSubscriberCallback(const person_identification::msg::PersonNames::SharedPtr msg);
    void videoAnalysisSubscriberCallback(const video_analyzer::msg::VideoAnalysis::SharedPtr msg);

    double personNameDistance(const person_identification::msg::PersonName& name);
    double faceDistance(
        const video_analyzer::msg::VideoAnalysisObject& object,
        const tf2::Stamped<tf2::Transform>& transform);
};

inline std::type_index SmartIdleState::type() const
{
    return std::type_index(typeid(SmartIdleState));
}

#endif
