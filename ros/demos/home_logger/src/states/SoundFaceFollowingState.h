#ifndef HOME_LOGGER_STATES_SOUND_FACE_FOLLOWING_STATE_H
#define HOME_LOGGER_STATES_SOUND_FACE_FOLLOWING_STATE_H

#include "State.h"

class SoundFaceFollowingState : public State
{
    std::type_index m_followingDesireType;
    uint64_t m_followingDesireId;
    uint64_t m_videoAnalysisWithoutPersonCount;

public:
    SoundFaceFollowingState(
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~SoundFaceFollowingState() override;

protected:
    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg) override;

private:
    void setFollowingDesire(std::unique_ptr<Desire> desire);
    bool containsAtLeastOnePerson(const video_analyzer::VideoAnalysis::ConstPtr& msg);
};

#endif
