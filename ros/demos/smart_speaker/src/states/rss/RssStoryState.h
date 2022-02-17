#ifndef SMART_SPEAKER_STATES_RSS_RSS_STORY_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_STORY_STATE_H

#include "../State.h"

#include <talk/Done.h>

#include <queue>

struct StoryLine
{
    std::string faceAnimation;
    std::string text;

    StoryLine(std::string faceAnimation, std::string text);
};

class RssStoryState : public State
{
    ros::Subscriber m_talkDoneSubscriber;

    std::queue<StoryLine> m_storyLines;
    uint64_t m_talkDesireId;
    uint64_t m_faceAnimationDesireId;

public:
    RssStoryState(Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        const std::string& englishStoryPath,
        const std::string& frenchStoryPath);
    ~RssStoryState() override = default;

    DECLARE_NOT_COPYABLE(RssStoryState);
    DECLARE_NOT_MOVABLE(RssStoryState);

protected:
    std::type_index type() const override;

    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void readStory(const std::string& storyPath);
    bool setNextLineDesire();
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index RssStoryState::type() const
{
    return std::type_index(typeid(RssStoryState));
}

#endif
