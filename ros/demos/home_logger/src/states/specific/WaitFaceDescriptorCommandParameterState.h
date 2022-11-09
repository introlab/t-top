#ifndef HOME_LOGGER_STATES_SPECIFIC_WAIT_FACE_DESCRIPTOR_COMMAND_PARAMETER_STATE_H
#define HOME_LOGGER_STATES_SPECIFIC_WAIT_FACE_DESCRIPTOR_COMMAND_PARAMETER_STATE_H

#include "../common/SoundFaceFollowingState.h"

#include <home_logger_common/commands/AllCommandParser.h>

class WaitFaceDescriptorCommandParameterStateParameter : public StateParameter
{
public:
    std::shared_ptr<Command> command;

    WaitFaceDescriptorCommandParameterStateParameter();
    explicit WaitFaceDescriptorCommandParameterStateParameter(std::shared_ptr<Command> command);
    ~WaitFaceDescriptorCommandParameterStateParameter() override;

    std::string toString() const override;
};

class WaitFaceDescriptorCommandParameterState : public SoundFaceFollowingState
{
    float m_noseConfidenceThreshold;

    WaitFaceDescriptorCommandParameterStateParameter m_parameter;
    std::vector<FaceDescriptor> m_faceDescriptors;

    std::optional<uint64_t> m_faceAnimationDesireId;
    std::optional<uint64_t> m_videoAnalyzer3dDesireId;

public:
    WaitFaceDescriptorCommandParameterState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        float noseConfidenceThreshold);
    ~WaitFaceDescriptorCommandParameterState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(WaitFaceDescriptorCommandParameterState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg) override;
    void onStateTimeout() override;

private:
    std::optional<FaceDescriptor> findNearestFaceDescriptor(const video_analyzer::VideoAnalysis::ConstPtr& msg);
    void switchState();
};

#endif
