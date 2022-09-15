#include "RssStoryState.h"

#include "../StateManager.h"

#include "../common/AfterTaskDelayState.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

#include <fstream>

using namespace std;

StoryLine::StoryLine(std::string faceAnimation, std::string text) : faceAnimation(faceAnimation), text(text) {}

RssStoryState::RssStoryState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    const std::string& englishStoryPath,
    const std::string& frenchStoryPath)
    : State(language, stateManager, desireSet, nodeHandle),
      m_talkDesireId(MAX_DESIRE_ID),
      m_faceAnimationDesireId(MAX_DESIRE_ID)
{
    m_desireSet->addObserver(this);

    switch (language)
    {
        case Language::ENGLISH:
            readStory(englishStoryPath);
            break;
        case Language::FRENCH:
            readStory(frenchStoryPath);
            break;
    }
}

RssStoryState::~RssStoryState()
{
    m_desireSet->removeObserver(this);
}

void RssStoryState::onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _)
{
    if (!enabled() || m_desireSet->contains(m_talkDesireId))
    {
        return;
    }

    bool finished = false;
    {
        auto transaction = m_desireSet->beginTransaction();
        finished = !setNextLineDesire();
    }

    if (finished)
    {
        m_stateManager.switchTo<AfterTaskDelayState>();
    }
}

void RssStoryState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    m_talkDesireId = MAX_DESIRE_ID;
    m_faceAnimationDesireId = MAX_DESIRE_ID;

    auto faceFollowingDesire = make_unique<NearestFaceFollowingDesire>();

    m_desireIds.emplace_back(faceFollowingDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(faceFollowingDesire));

    setNextLineDesire();
}

void RssStoryState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
    m_faceAnimationDesireId = MAX_DESIRE_ID;
}

void RssStoryState::readStory(const std::string& storyPath)
{
    ifstream storyFile(storyPath);
    string line;
    while (getline(storyFile, line))
    {
        size_t separatorPosition = line.find('|');
        if (separatorPosition == string::npos)
        {
            m_storyLines.emplace("", line);
        }
        else
        {
            string faceAnimation = trimString(line.substr(0, separatorPosition));
            string text = trimString(line.substr(separatorPosition + 1));
            m_storyLines.emplace(move(faceAnimation), move(text));
        }
    }

    m_storyLines.emplace("normal", "");
}

bool RssStoryState::setNextLineDesire()
{
    if (m_storyLines.empty())
    {
        return false;
    }

    if (m_talkDesireId != MAX_DESIRE_ID)
    {
        m_desireSet->removeDesire(m_talkDesireId);
        m_desireIds.pop_back();
    }
    if (m_faceAnimationDesireId != MAX_DESIRE_ID)
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId);
        m_desireIds.pop_back();
    }

    auto talkDesire = make_unique<TalkDesire>(m_storyLines.front().text);
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>(m_storyLines.front().faceAnimation);
    m_storyLines.pop();

    m_talkDesireId = talkDesire->id();
    m_faceAnimationDesireId = faceAnimationDesire->id();

    m_desireIds.emplace_back(talkDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    m_desireSet->addDesire(move(talkDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));

    return true;
}
