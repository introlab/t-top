#include "SmartWaitAnswerState.h"
#include "SmartIdleState.h"
#include "SmartValidTaskState.h"

#include "../StateManager.h"
#include "../InvalidTaskState.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

#include <unordered_set>

using namespace std;

static const string ENGLISH_WEATHER_WORD = "weather";
static const string ENGLISH_DANCE_WORD = "dance";

static const string FRENCH_WEATHER_WORD = "météo";
static const string FRENCH_DANCE_WORD = "danse";

SmartWaitAnswerState::SmartWaitAnswerState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    vector<vector<string>> songKeywords)
    : WaitAnswerState(language, stateManager, desireSet, nodeHandle),
      m_songKeywords(move(songKeywords)),
      m_randomGenerator(random_device()()),
      m_songIndexDistribution(0, m_songKeywords.size() - 1)
{
    if (m_songKeywords.size() == 0)
    {
        throw runtime_error("songKeywords must not be empty");
    }

    switch (language)
    {
        case Language::ENGLISH:
            m_weatherWord = ENGLISH_WEATHER_WORD;
            m_danceWord = ENGLISH_DANCE_WORD;
            break;
        case Language::FRENCH:
            m_weatherWord = FRENCH_WEATHER_WORD;
            m_danceWord = FRENCH_DANCE_WORD;
            break;
    }
}

void SmartWaitAnswerState::switchStateAfterTranscriptReceived(const std::string& text)
{
    auto lowerCaseText = toLowerString(text);

    // TODO Improve the task classification
    bool weather = lowerCaseText.find(m_weatherWord) != string::npos;
    bool dance = lowerCaseText.find(m_danceWord) != string::npos;
    size_t songIndex = getSongIndex(lowerCaseText);


    if (weather && !dance)
    {
        m_stateManager.switchTo<SmartValidTaskState>(CURRENT_WEATHER_TASK);
    }
    else if (!weather && dance && songIndex != string::npos)
    {
        m_stateManager.switchTo<SmartValidTaskState>(string(DANCE_TASK) + '|' + to_string(songIndex));
    }
    else
    {
        ROS_WARN_STREAM("Invalid task (" << text << ")");
        m_stateManager.switchTo<InvalidTaskState>();
    }
}

void SmartWaitAnswerState::switchStateAfterTimeout()
{
    m_stateManager.switchTo<SmartIdleState>();
}

size_t SmartWaitAnswerState::getSongIndex(const std::string& text)
{
    size_t songIndex = string::npos;

    for (size_t i = 0; i < m_songKeywords.size(); i++)
    {
        size_t isSongSelected = containsAllKeywords(text, m_songKeywords[i]);
        if (isSongSelected && songIndex == string::npos)
        {
            songIndex = i;
        }
        else if (isSongSelected && songIndex != string::npos)
        {
            return string::npos;
        }
    }

    if (songIndex == string::npos)
    {
        songIndex = m_songIndexDistribution(m_randomGenerator);
    }

    return songIndex;
}

bool SmartWaitAnswerState::containsAllKeywords(const string& text, const vector<string>& keywords)
{
    size_t count = 0;
    for (size_t i = 0; i < keywords.size(); i++)
    {
        size_t songNameIndex = text.find(keywords[i]);
        if (songNameIndex != string::npos)
        {
            count++;
        }
    }

    return count == keywords.size();
}
