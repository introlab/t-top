#include "SmartIdleState.h"
#include "SmartAskTaskState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

SmartIdleState::SmartIdleState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    double personDistanceThreshold,
    std::string personDistanceFrameId,
    double noseConfidenceThreshold,
    size_t videoAnalysisMessageCountThreshold,
    size_t videoAnalysisMessageCountTolerance)
    : State(language, stateManager, desireSet, nodeHandle),
      m_personDistanceThreshold(personDistanceThreshold),
      m_personDistanceFrameId(personDistanceFrameId),
      m_noseConfidenceThreshold(noseConfidenceThreshold),
      m_videoAnalysisMessageCountThreshold(videoAnalysisMessageCountThreshold),
      m_videoAnalysisMessageCountTolerance(videoAnalysisMessageCountTolerance),
      m_videoAnalysisValidMessageCount(0),
      m_videoAnalysisInvalidMessageCount(0)
{
    m_personNamesSubscriber =
        nodeHandle.subscribe("person_names", 1, &SmartIdleState::personNamesSubscriberCallback, this);

    m_videoAnalysisSubscriber =
        nodeHandle.subscribe("video_analysis", 1, &SmartIdleState::videoAnalysisSubscriberCallback, this);
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

void SmartIdleState::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
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

void SmartIdleState::videoAnalysisSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    if (!enabled() || msg->objects.size() == 0)
    {
        return;
    }
    if (!msg->contains_3d_positions)
    {
        ROS_ERROR("The video analysis must contain 3d positions.");
        return;
    }

    tf::StampedTransform transform;
    try
    {
        m_tfListener.lookupTransform(m_personDistanceFrameId, msg->header.frame_id, msg->header.stamp, transform);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR("%s", ex.what());
        return;
    }

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

double SmartIdleState::personNameDistance(const person_identification::PersonName& name)
{
    if (name.position_3d.size() == 0)
    {
        return numeric_limits<double>::infinity();
    }

    try
    {
        tf::StampedTransform transform;
        m_tfListener.lookupTransform(m_personDistanceFrameId, name.frame_id, ros::Time(0), transform);

        tf::Vector3 p(name.position_3d[0].x, name.position_3d[0].y, name.position_3d[0].z);
        p = transform * p;
        return p.length();
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR("%s", ex.what());
        return numeric_limits<double>::infinity();
    }
}

double SmartIdleState::faceDistance(
    const video_analyzer::VideoAnalysisObject& object,
    const tf::StampedTransform& transform)
{
    constexpr size_t PERSON_POSE_NOSE_INDEX = 0;

    if (object.face_descriptor.size() == 0 ||
        object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < m_noseConfidenceThreshold ||
        object.person_pose_3d.size() == 0)
    {
        return numeric_limits<double>::infinity();
    }

    auto nosePoint = object.person_pose_3d[PERSON_POSE_NOSE_INDEX];
    tf::Vector3 p(nosePoint.x, nosePoint.y, nosePoint.z);
    p = transform * p;
    return p.length();
}
