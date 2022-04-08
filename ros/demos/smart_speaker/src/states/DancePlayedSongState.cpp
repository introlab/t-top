#include "DancePlayedSongState.h"
#include "StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

DancePlayedSongState::DancePlayedSongState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType,
    vector<std::string> songPaths)
    : State(language, stateManager, desireSet, nodeHandle),
      m_nextStateType(nextStateType),
      m_songPaths(move(songPaths)),
      m_songDesireId(MAX_DESIRE_ID)
{
    if (m_songPaths.size() == 0)
    {
        throw runtime_error("songPaths must not be empty");
    }

    m_songStartedSubscriber =
        nodeHandle.subscribe("sound_player/started", 1, &DancePlayedSongState::songStartedSubscriberCallback, this);
    m_songDoneSubscriber =
        nodeHandle.subscribe("sound_player/done", 1, &DancePlayedSongState::songDoneSubscriberCallback, this);
}

void DancePlayedSongState::enable(const string& parameter)
{
    State::enable(parameter);

    size_t songIndex = atoi(parameter.c_str());
    if (songIndex >= m_songPaths.size())
    {
        ROS_ERROR("The song index is invalid.");
        m_stateManager.switchTo(m_nextStateType);
    }
    else
    {
        auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");
        auto songDesire = make_unique<PlaySoundDesire>(m_songPaths[songIndex]);
        m_songDesireId = songDesire->id();

        m_desireIds.emplace_back(faceAnimationDesire->id());
        m_desireIds.emplace_back(songDesire->id());

        auto transaction = m_desireSet->beginTransaction();
        m_desireSet->addDesire(move(faceAnimationDesire));
        m_desireSet->addDesire(move(songDesire));
    }
}

void DancePlayedSongState::disable()
{
    State::disable();
    m_songDesireId = MAX_DESIRE_ID;
}

void DancePlayedSongState::songStartedSubscriberCallback(const sound_player::Started::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_songDesireId)
    {
        return;
    }

    auto danceDesire = make_unique<DanceDesire>();
    m_desireIds.emplace_back(danceDesire->id());
    m_desireSet->addDesire(move(danceDesire));
}

void DancePlayedSongState::songDoneSubscriberCallback(const sound_player::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_songDesireId)
    {
        return;
    }
    if (!msg->ok)
    {
        ROS_ERROR("Unable to dance the played song");
    }

    m_stateManager.switchTo(m_nextStateType);
}
