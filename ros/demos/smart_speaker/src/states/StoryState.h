#ifndef SMART_SPEAKER_STATES_STORY_STATE_H
#define SMART_SPEAKER_STATES_STORY_STATE_H

#include "State.h"

#include <talk/Done.h>

#include <queue>

struct StoryLine
{
    std::string faceAnimation;
    std::string text;

    StoryLine(std::string faceAnimation, std::string text);
};

class StoryState : public State
{
    ros::Subscriber m_talkDoneSubscriber;

    std::queue<StoryLine> m_storyLines;
    uint64_t m_talkDesireId;
    uint64_t m_faceAnimationDesireId;

public:
    StoryState(StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle,
        const std::string& storyPath);
    ~StoryState() override = default;

    DECLARE_NOT_COPYABLE(StoryState);
    DECLARE_NOT_MOVABLE(StoryState);

    std::type_index type() const override;

protected:
    void enable(const std::string& parameter) override;
    void disable() override;

private:
    void readStory(const std::string& storyPath);
    bool setNextLineDesire();
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

inline std::type_index StoryState::type() const
{
    return std::type_index(typeid(StoryState));
}

#endif
