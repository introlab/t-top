#include "SmartIdleState.h"
#include "SmartAskTaskState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

SmartIdleState::SmartIdleState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    double personDistanceThreshold,
    std::string personDistanceFrameId,
    double noseConfidenceThreshold,
    size_t videoAnalysisMessageCountThreshold,
    size_t videoAnalysisMessageCountTolerance)
    : State(language, stateManager, desireSet, move(node)),
      m_personDistanceThreshold(personDistanceThreshold),
      m_personDistanceFrameId(personDistanceFrameId),
      m_noseConfidenceThreshold(noseConfidenceThreshold),
      m_videoAnalysisMessageCountThreshold(videoAnalysisMessageCountThreshold),
      m_videoAnalysisMessageCountTolerance(videoAnalysisMessageCountTolerance),
      m_videoAnalysisValidMessageCount(0),
      m_videoAnalysisInvalidMessageCount(0)
{
    m_personNamesSubscriber = m_node->create_subscription<perception_msgs::msg::PersonNames>(
        "person_names",
        1,
        [this](const perception_msgs::msg::PersonNames::SharedPtr msg) { personNamesSubscriberCallback(msg); });

    m_videoAnalysisSubscriber = m_node->create_subscription<perception_msgs::msg::VideoAnalysis>(
        "video_analysis",
        1,
        [this](const perception_msgs::msg::VideoAnalysis::SharedPtr msg) { videoAnalysisSubscriberCallback(msg); });

    m_tfBuffer = make_unique<tf2_ros::Buffer>(m_node->get_clock());
    m_tfListener = make_shared<tf2_ros::TransformListener>(*m_tfBuffer);
}

void SmartIdleState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto videoAnalyzerDesire = make_unique<FastVideoAnalyzer3dDesire>();
    auto gestureDesire = make_unique<GestureDesire>("origin_all");
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(videoAnalyzerDesire->id());
    m_desireIds.emplace_back(gestureDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(videoAnalyzerDesire));
    m_desireSet->addDesire(move(gestureDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
}

void SmartIdleState::personNamesSubscriberCallback(const perception_msgs::msg::PersonNames::SharedPtr msg)
{
    if (!enabled() || msg->names.size() == 0)
    {
        return;
    }

    for (auto& name : msg->names)
    {
        double distance = personNameDistance(name);
        if (distance <= m_personDistanceThreshold)
        {
            m_stateManager.switchTo<SmartAskTaskState>(name.name);
            break;
        }
    }
}

void SmartIdleState::videoAnalysisSubscriberCallback(const perception_msgs::msg::VideoAnalysis::SharedPtr msg)
{
    if (!enabled() || msg->objects.size() == 0)
    {
        return;
    }
    if (!msg->contains_3d_positions)
    {
        RCLCPP_ERROR(m_node->get_logger(), "The video analysis must contain 3d positions.");
        return;
    }

    geometry_msgs::msg::TransformStamped transformMsg;
    try
    {
        transformMsg = m_tfBuffer->lookupTransform(m_personDistanceFrameId, msg->header.frame_id, msg->header.stamp);
    }
    catch (tf2::TransformException& ex)
    {
        RCLCPP_ERROR(m_node->get_logger(), "%s", ex.what());
        return;
    }
    tf2::Stamped<tf2::Transform> transform;
    tf2::convert(transformMsg, transform);

    bool faceFound = false;
    for (auto& object : msg->objects)
    {
        double distance = faceDistance(object, transform);
        if (distance <= m_personDistanceThreshold)
        {
            m_videoAnalysisValidMessageCount++;
            m_videoAnalysisInvalidMessageCount = 0;
            faceFound = true;
            break;
        }
    }

    if (!faceFound)
    {
        m_videoAnalysisInvalidMessageCount++;
    }
    if (m_videoAnalysisInvalidMessageCount > m_videoAnalysisMessageCountTolerance)
    {
        m_videoAnalysisValidMessageCount = 0;
    }
    if (m_videoAnalysisValidMessageCount >= m_videoAnalysisMessageCountThreshold)
    {
        m_stateManager.switchTo<SmartAskTaskState>();
    }
}

double SmartIdleState::personNameDistance(const perception_msgs::msg::PersonName& name)
{
    if (name.position_3d.size() == 0)
    {
        return numeric_limits<double>::infinity();
    }

    try
    {
        geometry_msgs::msg::TransformStamped transformMsg =
            m_tfBuffer->lookupTransform(m_personDistanceFrameId, name.frame_id, tf2::TimePointZero);
        tf2::Stamped<tf2::Transform> transform;
        tf2::convert(transformMsg, transform);

        tf2::Vector3 p(name.position_3d[0].x, name.position_3d[0].y, name.position_3d[0].z);
        p = transform * p;
        return p.length();
    }
    catch (tf2::TransformException& ex)
    {
        RCLCPP_ERROR(m_node->get_logger(), "%s", ex.what());
        return numeric_limits<double>::infinity();
    }
}

double SmartIdleState::faceDistance(
    const perception_msgs::msg::VideoAnalysisObject& object,
    const tf2::Stamped<tf2::Transform>& transform)
{
    constexpr size_t PERSON_POSE_NOSE_INDEX = 0;

    if (object.face_descriptor.size() == 0 ||
        object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < m_noseConfidenceThreshold ||
        object.person_pose_3d.size() == 0)
    {
        return numeric_limits<double>::infinity();
    }

    auto nosePoint = object.person_pose_3d[PERSON_POSE_NOSE_INDEX];
    tf2::Vector3 p(nosePoint.x, nosePoint.y, nosePoint.z);
    p = transform * p;
    return p.length();
}
