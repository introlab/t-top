#ifndef HOME_LOGGER_STATES_COMMON_SOUND_FACE_FOLLOWING_STATE_H
#define HOME_LOGGER_STATES_COMMON_SOUND_FACE_FOLLOWING_STATE_H

#include "../State.h"

#include <optional>

class SoundFaceFollowingState : public State
{
    DesireType m_followingDesireType;
    std::optional<uint64_t> m_followingDesireId;
    uint64_t m_videoAnalysisWithoutPersonCount;

public:
    SoundFaceFollowingState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        rclcpp::Node::SharedPtr node);
    ~SoundFaceFollowingState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(SoundFaceFollowingState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onVideoAnalysisReceived(const perception_msgs::msg::VideoAnalysis::SharedPtr& msg) override;

private:
    void setFollowingDesire(std::unique_ptr<Desire> desire);
};

#endif
