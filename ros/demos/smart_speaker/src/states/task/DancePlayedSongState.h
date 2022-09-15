#ifndef SMART_SPEAKER_STATES_TASK_DANCE_PLAYED_SONG_STATE_H
#define SMART_SPEAKER_STATES_TASK_DANCE_PLAYED_SONG_STATE_H

#include "../State.h"

#include <sound_player/Started.h>

class DancePlayedSongState : public State, public DesireSetObserver
{
    std::type_index m_nextStateType;

    ros::Subscriber m_songStartedSubscriber;

    std::vector<std::string> m_songPaths;
    uint64_t m_songDesireId;

public:
    DancePlayedSongState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType,
        std::vector<std::string> songPaths);
    ~DancePlayedSongState() override;

    DECLARE_NOT_COPYABLE(DancePlayedSongState);
    DECLARE_NOT_MOVABLE(DancePlayedSongState);

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter, const std::type_index& previousStageType) override;
    void disable() override;

private:
    void songStartedSubscriberCallback(const sound_player::Started::ConstPtr& msg);
};

inline std::type_index DancePlayedSongState::type() const
{
    return std::type_index(typeid(DancePlayedSongState));
}

#endif
