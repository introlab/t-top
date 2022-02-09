#include "RssWaitAnswerState.h"
#include "RssIdleState.h"
#include "RssValidTaskState.h"

#include "../StateManager.h"
#include "../InvalidTaskState.h"

#include "../../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

#include <unordered_set>

using namespace std;

static const string ENGLISH_WEATHER_WORD = "weather";
static const string ENGLISH_FORECAST_WORD = "forecast";
static const string ENGLISH_STORY_WORD = "story";
static const string ENGLISH_DANCE_WORD = "dance";
static const string ENGLISH_SONG_WORD = "song";

static const string FRENCH_WEATHER_WORD = "météo";
static const string FRENCH_FORECAST_WORD = "prévisions";
static const string FRENCH_STORY_WORD = "histoire";
static const string FRENCH_DANCE_WORD = "danses";
static const string FRENCH_SONG_WORD = "chanson";

RssWaitAnswerState::RssWaitAnswerState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(language, stateManager, desireSet, nodeHandle)
{
    m_speechToTextSubscriber = nodeHandle.subscribe("speech_to_text/transcript", 1,
        &RssWaitAnswerState::speechToTextSubscriberCallback, this);

    switch (language)
    {
    case Language::ENGLISH:
        m_weatherWord = ENGLISH_WEATHER_WORD;
        m_forecastWord = ENGLISH_FORECAST_WORD;
        m_storyWord = ENGLISH_STORY_WORD;
        m_danceWord = ENGLISH_DANCE_WORD;
        m_songWord = ENGLISH_SONG_WORD;
        break;
    case Language::FRENCH:
        m_weatherWord = FRENCH_WEATHER_WORD;
        m_forecastWord = FRENCH_FORECAST_WORD;
        m_storyWord = FRENCH_STORY_WORD;
        m_danceWord = FRENCH_DANCE_WORD;
        m_songWord = FRENCH_SONG_WORD;
        break;
    }
}

void RssWaitAnswerState::enable(const string& parameter)
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
        &RssWaitAnswerState::timeoutTimerCallback, this, oneshot);
}

void RssWaitAnswerState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void RssWaitAnswerState::speechToTextSubscriberCallback(const std_msgs::String::ConstPtr& msg)
{
    if (!enabled())
    {
        return;
    }

    auto words = splitStrings(toLowerString(msg->data), " \n.,!?");
    unordered_set<string> wordSet(words.begin(), words.end());

    // TODO Improve the task classification
    bool weather = static_cast<bool>(wordSet.count(m_weatherWord));
    bool forecast = static_cast<bool>(wordSet.count(m_forecastWord));
    bool story = static_cast<bool>(wordSet.count(m_storyWord));
    bool dance = static_cast<bool>(wordSet.count(m_danceWord));
    bool song = static_cast<bool>(wordSet.count(m_songWord));

    if (weather && !forecast && !story && !dance && !song)
    {
        m_stateManager.switchTo<RssValidTaskState>(CURRENT_WEATHER_TASK);
    }
    else if (forecast && !story && !dance && !song)
    {
        m_stateManager.switchTo<RssValidTaskState>(WEATHER_FORECAST_TASK);
    }
    else if (!weather && !forecast && story && !dance && !song)
    {
        m_stateManager.switchTo<RssValidTaskState>(STORY_TASK);
    }
    else if (!weather && !forecast && !story && dance && !song)
    {
        m_stateManager.switchTo<RssValidTaskState>(DANCE_TASK);
    }
    else if (!weather && !forecast && !story && dance && song)
    {
        m_stateManager.switchTo<RssValidTaskState>(DANCE_PLAYED_SONG_TASK);
    }
    else
    {
        m_stateManager.switchTo<InvalidTaskState>();
    }
}

void RssWaitAnswerState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<RssIdleState>();
}
