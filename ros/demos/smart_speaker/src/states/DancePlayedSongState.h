#ifndef SMART_SPEAKER_STATES_DANCE_PLAYED_SONG_STATE_H
#define SMART_SPEAKER_STATES_DANCE_PLAYED_SONG_STATE_H

#include "State.h"

#include <sound_player/Started.h>
#include <sound_player/Done.h>

class DancePlayedSongState : public State
{
    std::type_index m_nextStateType;

    ros::Subscriber m_songStartedSubscriber;
    ros::Subscriber m_songDoneSubscriber;

    std::vector<std::string> m_songPaths;
    uint64_t m_songDesireId;

public:
    DancePlayedSongState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::type_index nextStateType,
        std::vector<std::string> songPaths);
    ~DancePlayedSongState() override = default;

    DECLARE_NOT_COPYABLE(DancePlayedSongState);
    DECLARE_NOT_MOVABLE(DancePlayedSongState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void songStartedSubscriberCallback(const sound_player::Started::ConstPtr& msg);
    void songDoneSubscriberCallback(const sound_player::Done::ConstPtr& msg);
};

inline std::type_index DancePlayedSongState::type() const
{
    return std::type_index(typeid(DancePlayedSongState));
}

#endif
