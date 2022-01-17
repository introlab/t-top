#include <t_top/hbba_lite/Strategies.h>

#include <std_msgs/String.h>
#include <talk/Text.h>
#include <gesture/GestureName.h>
#include <sound_player/SoundFile.h>

using namespace std;

FaceAnimationStrategy::FaceAnimationStrategy(uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<FaceAnimationDesire>(utility, {}, {}, move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_animationPublisher = nodeHandle.advertise<std_msgs::String>("face/animation", 1);
}

void FaceAnimationStrategy::onEnabling(const unique_ptr<Desire>& desire)
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

TalkStrategy::TalkStrategy(uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<TalkDesire>(utility,
            {{"sound", 1}},
            {{"talk/filter_state", FilterConfiguration::onOff()}},
            move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_talkPublisher = nodeHandle.advertise<talk::Text>("talk/text", 1);
}

void TalkStrategy::onEnabling(const unique_ptr<Desire>& desire)
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

GestureStrategy::GestureStrategy(uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<GestureDesire>(utility,
            {{"motor", 1}},
            {{"gesture/filter_state", FilterConfiguration::onOff()}},
            move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_gesturePublisher = nodeHandle.advertise<gesture::GestureName>("gesture/name", 1);
}

void GestureStrategy::onEnabling(const unique_ptr<Desire>& desire)
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

PlaySoundStrategy::PlaySoundStrategy(uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle) :
        Strategy<PlaySoundDesire>(utility,
            {{"sound", 1}},
            {{"sound_player/filter_state", FilterConfiguration::onOff()}},
            move(filterPool)),
        m_nodeHandle(nodeHandle)
{
    m_pathPublisher = nodeHandle.advertise<sound_player::SoundFile>("sound_player/file", 1);
}

void PlaySoundStrategy::onEnabling(const unique_ptr<Desire>& desire)
{
    Strategy<PlaySoundDesire>::onEnabling(desire);

    auto playSoundDesire = dynamic_cast<PlaySoundDesire*>(desire.get());
    if (playSoundDesire != nullptr)
    {
        sound_player::SoundFile msg;
        msg.path = playSoundDesire->path();
        msg.id = playSoundDesire->id();
        m_pathPublisher.publish(msg);
    }
}


unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<RobotNameDetectorDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"robot_name_detector/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createSlowVideoAnalyzerStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SlowVideoAnalyzerDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(15)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzerStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzerDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(3)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzerWithAnalyzedImageStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzerWithAnalyzedImageDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(3)},
            {"video_analyzer/analysed_image/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<AudioAnalyzerDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"audio_analyzer/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createSpeechToTextStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SpeechToTextDesire>>(utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"speech_to_text/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}


unique_ptr<BaseStrategy> createExploreStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<ExploreDesire>>(utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{{"explore/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createFaceAnimationStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<FaceAnimationStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createSoundFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SoundFollowingDesire>>(utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{{"sound_following/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createFaceFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FaceFollowingDesire>>(utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>
        {
            {"video_analyzer/image_raw/filter_state", FilterConfiguration::throttling(3)},
            {"face_following/filter_state", FilterConfiguration::onOff()}
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createTalkStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<TalkStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createGestureStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<GestureStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createDanceStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<DanceDesire>>(utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>
        {
            {"beat_detector/filter_state", FilterConfiguration::onOff()},
            {"head_dance/filter_state", FilterConfiguration::onOff()},
            {"torso_dance/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createPlaySoundStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle, uint16_t utility)
{
    return make_unique<PlaySoundStrategy>(utility, move(filterPool), nodeHandle);
}
