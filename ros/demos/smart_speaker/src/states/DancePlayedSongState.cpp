#include "DancePlayedSongState.h"
#include "StateManager.h"

#include <t_top/hbba_lite/Desires.h>

using namespace std;

DancePlayedSongState::DancePlayedSongState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    type_index nextStateType,
    string songPath) :
        State(language, stateManager, desireSet, nodeHandle),
        m_nextStateType(nextStateType),
        m_songPath(move(songPath)),
        m_songDesireId(MAX_DESIRE_ID)
{
    m_songStartedSubscriber = nodeHandle.subscribe("sound_player/started", 1,
        &DancePlayedSongState::songStartedSubscriberCallback, this);
    m_songDoneSubscriber = nodeHandle.subscribe("sound_player/done", 1,
        &DancePlayedSongState::songDoneSubscriberCallback, this);
}

void DancePlayedSongState::enable(const string& parameter)
{
    State::enable(parameter);

    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");
    auto songDesire = make_unique<PlaySoundDesire>(m_songPath);
    m_songDesireId = songDesire->id();

    m_desireIds.emplace_back(faceAnimationDesire->id());
    m_desireIds.emplace_back(songDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(faceAnimationDesire));
    m_desireSet->addDesire(move(songDesire));
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

    m_stateManager.switchTo(m_nextStateType);
}
