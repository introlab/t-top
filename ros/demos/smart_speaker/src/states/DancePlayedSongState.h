#ifndef SMART_SPEAKER_STATES_DANCE_PLAYED_SONG_STATE_H
#define SMART_SPEAKER_STATES_DANCE_PLAYED_SONG_STATE_H

#include "State.h"

#include <sound_player/Done.h>

class DancePlayedSongState : public State
{
    ros::Subscriber m_songDoneSubscriber;
    std::string m_songPath;
    uint64_t m_songDesireId;

public:
    DancePlayedSongState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        std::string songPath);
    ~DancePlayedSongState() override = default;

    DECLARE_NOT_COPYABLE(DancePlayedSongState);
    DECLARE_NOT_MOVABLE(DancePlayedSongState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void songDoneSubscriberCallback(const sound_player::Done::ConstPtr& msg);
};

inline std::type_index DancePlayedSongState::type() const
{
    return std::type_index(typeid(DancePlayedSongState));
}

#endif
