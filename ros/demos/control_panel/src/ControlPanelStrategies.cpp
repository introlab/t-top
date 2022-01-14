#include "ControlPanelStrategies.h"

#include <std_msgs/String.h>
#include <talk/Text.h>
#include <gesture/GestureName.h>

using namespace std;

FaceAnimationStrategy::FaceAnimationStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<FaceAnimationDesire>(1, {}, {}, move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_animationPublisher = nodeHandle.advertise<std_msgs::String>("face/animation", 1);
}

void FaceAnimationStrategy::onEnabling(const std::unique_ptr<Desire>& desire)
{
    Strategy<FaceAnimationDesire>::onEnabling(desire);

    auto faceAnimationDesire = dynamic_cast<FaceAnimationDesire*>(desire.get());
    if (faceAnimationDesire != nullptr)
    {
        std_msgs::String msg;
        msg.data = faceAnimationDesire->name();
        m_animationPublisher.publish(msg);
    }
}

TalkStrategy::TalkStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<TalkDesire>(1,
            {{"sound", 1}},
            {{"talk/filter_state", FilterConfiguration::onOff()}},
            move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_talkPublisher = nodeHandle.advertise<talk::Text>("talk/text", 1);
}

void TalkStrategy::onEnabling(const std::unique_ptr<Desire>& desire)
{
    Strategy<TalkDesire>::onEnabling(desire);

    auto talkDesire = dynamic_cast<TalkDesire*>(desire.get());
    if (talkDesire != nullptr)
    {
        talk::Text msg;
        msg.text = talkDesire->text();
        msg.id = talkDesire->id();
        m_talkPublisher.publish(msg);
    }
}

GestureStrategy::GestureStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<GestureDesire>(1,
            {{"motor", 1}},
            {{"gesture/filter_state", FilterConfiguration::onOff()}},
            move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_gesturePublisher = nodeHandle.advertise<gesture::GestureName>("gesture/name", 1);
}

void GestureStrategy::onEnabling(const std::unique_ptr<Desire>& desire)
{
    Strategy<GestureDesire>::onEnabling(desire);

    auto gestureDesire = dynamic_cast<GestureDesire*>(desire.get());
    if (gestureDesire != nullptr)
    {
        gesture::GestureName msg;
        msg.name = gestureDesire->name();
        msg.id = gestureDesire->id();
        m_gesturePublisher.publish(msg);
    }
}

unique_ptr<BaseStrategy> createFaceAnimationStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle)
{
    return make_unique<FaceAnimationStrategy>(move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createTalkStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle)
{
    return make_unique<TalkStrategy>(move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createListenStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<ListenDesire>>(1,
        unordered_map<std::string, uint16_t>{},
        unordered_map<std::string, FilterConfiguration>{{"speech_to_text/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createGestureStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle)
{
    return make_unique<GestureStrategy>(move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createFaceFollowingStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<FaceFollowingDesire>>(1,
        unordered_map<std::string, uint16_t>{{"motor", 1}},
        unordered_map<std::string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(2)},
            {"face_following/filter_state", FilterConfiguration::onOff()}
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createSoundFollowingStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<SoundFollowingDesire>>(1,
        unordered_map<std::string, uint16_t>{{"motor", 1}},
        unordered_map<std::string, FilterConfiguration>{{"sound_following/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createDanceStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<DanceDesire>>(1,
        unordered_map<std::string, uint16_t>{{"motor", 1}},
        unordered_map<std::string, FilterConfiguration>
        {
            {"beat_detector/filter_state", FilterConfiguration::onOff()},
            {"explore/filter_state", FilterConfiguration::onOff()}
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createExploreStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<ExploreDesire>>(1,
        unordered_map<std::string, uint16_t>{{"motor", 1}},
        unordered_map<std::string, FilterConfiguration>{{"explore/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createVideoAnalyzerStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<VideoAnalyzerDesire>>(1,
        unordered_map<std::string, uint16_t>{},
        unordered_map<std::string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(2)},
            {"video_analyzer/analysed_image/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<AudioAnalyzerDesire>>(1,
        unordered_map<std::string, uint16_t>{},
        unordered_map<std::string, FilterConfiguration>{{"audio_analyzer/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(shared_ptr<FilterPool> filterPool)
{
    return make_unique<Strategy<RobotNameDetectorDesire>>(1,
        unordered_map<std::string, uint16_t>{},
        unordered_map<std::string, FilterConfiguration>{{"robot_name_detector/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}
