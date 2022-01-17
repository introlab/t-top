#include "StoryState.h"
#include "StateManager.h"
#include "IdleState.h"

#include "../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

#include <fstream>

using namespace std;

StoryLine::StoryLine(std::string faceAnimation, std::string text) :
        faceAnimation(faceAnimation), text(text)
{
}

StoryState::StoryState(StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
        const std::string& storyPath) :
        State(stateManager, desireSet, nodeHandle),
        m_talkDesireId(MAX_DESIRE_ID),
        m_faceAnimationDesireId(MAX_DESIRE_ID)
{
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 1,
        &StoryState::talkDoneSubscriberCallback, this);

    readStory(storyPath);
}

void StoryState::enable(const string& parameter)
{
    State::enable(parameter);

    m_talkDesireId = MAX_DESIRE_ID;
    m_faceAnimationDesireId = MAX_DESIRE_ID;

    auto faceFollowingDesire = make_unique<FaceFollowingDesire>();

    m_desireIds.emplace_back(faceFollowingDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(faceFollowingDesire));

    setNextLineDesire();
}

void StoryState::disable()
{
    State::disable();
    m_talkDesireId = MAX_DESIRE_ID;
    m_faceAnimationDesireId = MAX_DESIRE_ID;
}

void StoryState::readStory(const std::string& storyPath)
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

bool StoryState::setNextLineDesire()
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

void StoryState::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (!enabled() || msg->id != m_talkDesireId)
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
        m_stateManager.switchTo<IdleState>();
    }
}
