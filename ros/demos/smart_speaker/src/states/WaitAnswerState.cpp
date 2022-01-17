#include "WaitAnswerState.h"
#include "StateManager.h"
#include "IdleState.h"
#include "ValidTaskState.h"
#include "InvalidTaskState.h"

#include "../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

#include <unordered_set>

using namespace std;

static const string WEATHER_WORD = "WEATHER";
static const string FORECAST_WORD = "FORECAST";
static const string STORY_WORD = "STORY";
static const string DANCE_WORD = "DANCE";
static const string SONG_WORD = "SONG";

WaitAnswerState::WaitAnswerState(StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(stateManager, desireSet, nodeHandle)
{
    m_speechToTextSubscriber = nodeHandle.subscribe("speech_to_text/transcript", 1,
        &WaitAnswerState::speechToTextSubscriberCallback, this);
}

void WaitAnswerState::enable(const string& parameter)
{
    State::enable(parameter);

    auto speechToTextDesire = make_unique<SpeechToTextDesire>();
    auto faceFollowingDesire = make_unique<FaceFollowingDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(speechToTextDesire->id());
    m_desireIds.emplace_back(faceFollowingDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(speechToTextDesire));
    m_desireSet->addDesire(move(faceFollowingDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));

    constexpr bool oneshot = true;
    m_timeoutTimer = m_nodeHandle.createTimer(ros::Duration(TIMEOUT_S),
        &WaitAnswerState::timeoutTimerCallback, this, oneshot);
}

void WaitAnswerState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void WaitAnswerState::speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg)
{
    if (!enabled())
    {
        return;
    }

    auto words = splitStrings(toUpperString(msg->data), " \n.,!?");
    unordered_set<string> wordSet(words.begin(), words.end());

    bool weather = static_cast<bool>(wordSet.count(WEATHER_WORD));
    bool forecast = static_cast<bool>(wordSet.count(FORECAST_WORD));
    bool story = static_cast<bool>(wordSet.count(STORY_WORD));
    bool dance = static_cast<bool>(wordSet.count(DANCE_WORD));
    bool song = static_cast<bool>(wordSet.count(SONG_WORD));

    if (weather && !forecast && !story && !dance && !song)
    {
        m_stateManager.switchTo<ValidTaskState>(CURRENT_WEATHER_TASK);
    }
    else if (forecast && !story && !dance && !song)
    {
        m_stateManager.switchTo<ValidTaskState>(WEATHER_FORECAST_TASK);
    }
    else if (!weather && !forecast && story && !dance && !song)
    {
        m_stateManager.switchTo<ValidTaskState>(STORY_TASK);
    }
    else if (!weather && !forecast && !story && dance && !song)
    {
        m_stateManager.switchTo<ValidTaskState>(DANCE_TASK);
    }
    else if (!weather && !forecast && !story && dance && song)
    {
        m_stateManager.switchTo<ValidTaskState>(DANCE_PLAYED_SONG_TASK);
    }
    else
    {
        m_stateManager.switchTo<InvalidTaskState>();
    }
}

void WaitAnswerState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<IdleState>();
}
