#include "WaitFaceDescriptorCommandParameterState.h"
#include "ExecuteCommandState.h"
#include "IdleState.h"

#include <home_logger_common/language/StringResources.h>

#include <t_top_hbba_lite/Desires.h>
#include <tf/transform_listener.h>

using namespace std;

constexpr size_t FACE_DESCRIPTOR_COUNT = 10;

WaitFaceDescriptorCommandParameterStateParameter::WaitFaceDescriptorCommandParameterStateParameter() {}

WaitFaceDescriptorCommandParameterStateParameter::WaitFaceDescriptorCommandParameterStateParameter(
    shared_ptr<Command> command)
    : command(move(command))
{
}

WaitFaceDescriptorCommandParameterStateParameter::~WaitFaceDescriptorCommandParameterStateParameter() {}

string WaitFaceDescriptorCommandParameterStateParameter::toString() const
{
    stringstream ss;
    ss << "command=" << command->type().name();
    return ss.str();
}


WaitFaceDescriptorCommandParameterState::WaitFaceDescriptorCommandParameterState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    float noseConfidenceThreshold)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_noseConfidenceThreshold(noseConfidenceThreshold)
{
}

WaitFaceDescriptorCommandParameterState::~WaitFaceDescriptorCommandParameterState() {}

void WaitFaceDescriptorCommandParameterState::onEnabling(
    const StateParameter& parameter,
    const StateType& previousStateType)
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_parameter = dynamic_cast<const WaitFaceDescriptorCommandParameterStateParameter&>(parameter);
    m_faceDescriptors.clear();

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("blink");
    if (!m_desireSet->containsAnyDesiresOfType<FastVideoAnalyzer3dDesire>())
    {
        m_videoAnalyzer3dDesireId = m_desireSet->addDesire<FastVideoAnalyzer3dDesire>();
    }
}

void WaitFaceDescriptorCommandParameterState::onDisabling()
{
    SoundFaceFollowingState::onDisabling();

    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = nullopt;
    }
    if (m_videoAnalyzer3dDesireId.has_value())
    {
        m_desireSet->removeDesire(m_videoAnalyzer3dDesireId.value());
        m_videoAnalyzer3dDesireId = nullopt;
    }
}

void WaitFaceDescriptorCommandParameterState::onVideoAnalysisReceived(
    const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    auto faceDescriptor = findNearestFaceDescriptor(msg);
    if (faceDescriptor.has_value())
    {
        m_faceDescriptors.emplace_back(move(faceDescriptor.value()));
    }

    if (m_faceDescriptors.size() >= FACE_DESCRIPTOR_COUNT)
    {
        switchState();
    }
}

void WaitFaceDescriptorCommandParameterState::onStateTimeout()
{
    switchState();
}

optional<FaceDescriptor> WaitFaceDescriptorCommandParameterState::findNearestFaceDescriptor(
    const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    constexpr size_t PERSON_POSE_NOSE_INDEX = 0;

    double nearestDistance = numeric_limits<double>::infinity();
    optional<FaceDescriptor> nearestFaceDescriptor;

    for (const auto& object : msg->objects)
    {
        if (object.face_descriptor.empty() || object.person_pose_confidence.size() <= PERSON_POSE_NOSE_INDEX ||
            object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < m_noseConfidenceThreshold ||
            object.person_pose_3d.size() <= PERSON_POSE_NOSE_INDEX)
        {
            continue;
        }

        auto nosePoint = object.person_pose_3d[PERSON_POSE_NOSE_INDEX];
        float distance = tf::Vector3(nosePoint.x, nosePoint.y, nosePoint.z).length();
        if (distance < nearestDistance)
        {
            nearestDistance = distance;
            nearestFaceDescriptor = FaceDescriptor(object.face_descriptor);
        }
    }

    return nearestFaceDescriptor;
}

void WaitFaceDescriptorCommandParameterState::switchState()
{
    if (m_faceDescriptors.size() < FACE_DESCRIPTOR_COUNT)
    {
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringResources::getValue("dialogs.wait_face_descriptor_command_parameter_state.timeout"),
            "no",
            "sad",
            StateType::get<IdleState>()));
    }
    else
    {
        m_stateManager.switchTo<ExecuteCommandState>(
            ExecuteCommandStateParameter(m_parameter.command, FaceDescriptor::mean(m_faceDescriptors)));
    }
}
