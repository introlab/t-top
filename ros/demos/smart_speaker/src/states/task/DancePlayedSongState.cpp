#include "DancePlayedSongState.h"

#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

DancePlayedSongState::DancePlayedSongState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    rclcpp::Node::SharedPtr node,
    type_index nextStateType,
    vector<std::string> songPaths)
    : State(language, stateManager, desireSet, move(node)),
      m_nextStateType(nextStateType),
      m_songPaths(move(songPaths)),
      m_songDesireId(MAX_DESIRE_ID)
{
    if (m_songPaths.size() == 0)
    {
        throw runtime_error("songPaths must not be empty");
    }

    m_desireSet->addObserver(this);

    m_songStartedSubscriber = m_node->create_subscription<behavior_msgs::msg::SoundStarted>(
        "sound_player/started",
        1,
        [this](const behavior_msgs::msg::SoundStarted::SharedPtr msg) { songStartedSubscriberCallback(msg); });
}

DancePlayedSongState::~DancePlayedSongState()
{
    m_desireSet->removeObserver(this);
}

void DancePlayedSongState::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    if (!enabled() || m_desireSet->contains(m_songDesireId))
    {
        return;
    }

    m_stateManager.switchTo(m_nextStateType);
}

void DancePlayedSongState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    size_t songIndex = atoi(parameter.c_str());
    if (songIndex >= m_songPaths.size())
    {
        RCLCPP_ERROR(m_node->get_logger(), "The song index is invalid.");
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

void DancePlayedSongState::songStartedSubscriberCallback(const behavior_msgs::msg::SoundStarted::SharedPtr msg)
{
    if (!enabled() || msg->id != m_songDesireId)
    {
        return;
    }

    auto danceDesire = make_unique<DanceDesire>();
    m_desireIds.emplace_back(danceDesire->id());
    m_desireSet->addDesire(move(danceDesire));
}
