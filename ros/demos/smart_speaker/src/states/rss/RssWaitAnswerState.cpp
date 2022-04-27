#include "RssWaitAnswerState.h"
#include "RssIdleState.h"
#include "RssValidTaskState.h"

#include "../StateManager.h"

#include "../common/InvalidTaskState.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

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

RssWaitAnswerState::RssWaitAnswerState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : WaitAnswerState(language, stateManager, desireSet, nodeHandle)
{
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

void RssWaitAnswerState::switchStateAfterTranscriptReceived(const std::string& text, bool isFinal)
{
    auto words = splitStrings(toLowerString(text), " \n.,!?");
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
    else if (isFinal)
    {
        m_stateManager.switchTo<InvalidTaskState>();
    }
}

void RssWaitAnswerState::switchStateAfterTimeout(bool transcriptReceived)
{
    if (transcriptReceived)
    {
        m_stateManager.switchTo<InvalidTaskState>();
    }
    else
    {
        m_stateManager.switchTo<RssIdleState>();
    }
}
