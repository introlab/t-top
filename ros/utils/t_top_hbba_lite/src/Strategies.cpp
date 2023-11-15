#include <t_top_hbba_lite/Strategies.h>

#include <std_msgs/String.h>
#include <talk/Text.h>
#include <gesture/GestureName.h>
#include <sound_player/SoundFile.h>

using namespace std;

FaceAnimationStrategy::FaceAnimationStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle)
    : Strategy<FaceAnimationDesire>(utility, {}, {}, move(filterPool)),
      m_nodeHandle(nodeHandle)
{
    m_animationPublisher = nodeHandle.advertise<std_msgs::String>("face/animation", 1);
}

StrategyType FaceAnimationStrategy::strategyType()
{
    return StrategyType::get<FaceAnimationStrategy>();
}

void FaceAnimationStrategy::onEnabling(const FaceAnimationDesire& desire)
{
    Strategy<FaceAnimationDesire>::onEnabling(desire);

    std_msgs::String msg;
    msg.data = desire.name();
    m_animationPublisher.publish(msg);
}

void FaceAnimationStrategy::onDisabling()
{
    std_msgs::String msg;
    msg.data = "normal";
    m_animationPublisher.publish(msg);

    Strategy<FaceAnimationDesire>::onDisabling();
}

LedEmotionStrategy::LedEmotionStrategy(uint16_t utility, shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle)
    : Strategy<LedEmotionDesire>(
          utility,
          {{"led", 1}},
          {{"led_emotions/filter_state", FilterConfiguration::onOff()}},
          move(filterPool)),
      m_nodeHandle(nodeHandle)
{
    m_emotionPublisher = nodeHandle.advertise<std_msgs::String>("led_emotions/name", 1);
}

StrategyType LedEmotionStrategy::strategyType()
{
    return StrategyType::get<LedEmotionStrategy>();
}

void LedEmotionStrategy::onEnabling(const LedEmotionDesire& desire)
{
    Strategy<LedEmotionDesire>::onEnabling(desire);

    std_msgs::String msg;
    msg.data = desire.name();
    m_emotionPublisher.publish(msg);
}

SpecificFaceFollowingStrategy::SpecificFaceFollowingStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle)
    : Strategy<SpecificFaceFollowingDesire>(
          utility,
          {{"motor", 1}},
          {{"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
           {"specific_face_following/filter_state", FilterConfiguration::onOff()}},
          move(filterPool)),
      m_nodeHandle(nodeHandle)
{
    m_targetNamePublisher = nodeHandle.advertise<std_msgs::String>("face_following/target_name", 1);
}

StrategyType SpecificFaceFollowingStrategy::strategyType()
{
    return StrategyType::get<SpecificFaceFollowingStrategy>();
}

void SpecificFaceFollowingStrategy::onEnabling(const SpecificFaceFollowingDesire& desire)
{
    Strategy<SpecificFaceFollowingDesire>::onEnabling(desire);

    std_msgs::String msg;
    msg.data = desire.targetName();
    m_targetNamePublisher.publish(msg);
}

TalkStrategy::TalkStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : Strategy<TalkDesire>(
          utility,
          {{"sound", 1}},
          {{"talk/filter_state", FilterConfiguration::onOff()}},
          move(filterPool)),
      m_desireSet(move(desireSet)),
      m_nodeHandle(nodeHandle)
{
    m_talkPublisher = nodeHandle.advertise<talk::Text>("talk/text", 1);
    m_talkDoneSubscriber = nodeHandle.subscribe("talk/done", 10, &TalkStrategy::talkDoneSubscriberCallback, this);
}

StrategyType TalkStrategy::strategyType()
{
    return StrategyType::get<TalkStrategy>();
}

void TalkStrategy::onEnabling(const TalkDesire& desire)
{
    Strategy<TalkDesire>::onEnabling(desire);

    talk::Text msg;
    msg.text = desire.text();
    msg.id = desire.id();
    m_talkPublisher.publish(msg);
}

void TalkStrategy::talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg)
{
    if (msg->id == desireId())
    {
        m_desireSet->removeDesire(msg->id);
    }
}

GestureStrategy::GestureStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : Strategy<GestureDesire>(
          utility,
          {{"motor", 1}},
          {{"gesture/filter_state", FilterConfiguration::onOff()}},
          move(filterPool)),
      m_desireSet(move(desireSet)),
      m_nodeHandle(nodeHandle)
{
    m_gesturePublisher = nodeHandle.advertise<gesture::GestureName>("gesture/name", 1);
    m_gestureDoneSubscriber =
        nodeHandle.subscribe("gesture/done", 1, &GestureStrategy::gestureDoneSubscriberCallback, this);
}

StrategyType GestureStrategy::strategyType()
{
    return StrategyType::get<GestureStrategy>();
}

void GestureStrategy::onEnabling(const GestureDesire& desire)
{
    Strategy<GestureDesire>::onEnabling(desire);

    gesture::GestureName msg;
    msg.name = desire.name();
    msg.id = desire.id();
    m_gesturePublisher.publish(msg);
}

void GestureStrategy::gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg)
{
    if (msg->id == desireId())
    {
        m_desireSet->removeDesire(msg->id);
    }
}

PlaySoundStrategy::PlaySoundStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : Strategy<PlaySoundDesire>(
          utility,
          {{"sound", 1}},
          {{"sound_player/filter_state", FilterConfiguration::onOff()}},
          move(filterPool)),
      m_desireSet(desireSet),
      m_nodeHandle(nodeHandle)
{
    m_pathPublisher = nodeHandle.advertise<sound_player::SoundFile>("sound_player/file", 1);
    m_soundDoneSubscriber =
        nodeHandle.subscribe("sound_player/done", 1, &PlaySoundStrategy::soundDoneSubscriberCallback, this);
}

StrategyType PlaySoundStrategy::strategyType()
{
    return StrategyType::get<GestureStrategy>();
}

void PlaySoundStrategy::onEnabling(const PlaySoundDesire& desire)
{
    Strategy<PlaySoundDesire>::onEnabling(desire);

    sound_player::SoundFile msg;
    msg.path = desire.path();
    msg.id = desire.id();
    m_pathPublisher.publish(msg);
}

void PlaySoundStrategy::soundDoneSubscriberCallback(const sound_player::Done::ConstPtr& msg)
{
    if (msg->id == desireId())
    {
        m_desireSet->removeDesire(msg->id);
    }
}


unique_ptr<BaseStrategy> createCamera3dRecordingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<Camera3dRecordingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_recorder_camera_3d/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createCamera2dWideRecordingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<Camera2dWideRecordingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_recorder_camera_2d_wide/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<RobotNameDetectorDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"robot_name_detector/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createSlowVideoAnalyzer3dStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SlowVideoAnalyzer3dDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(15)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzer3dStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer3dDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy>
    createFastVideoAnalyzer3dWithAnalyzedImageStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer3dWithAnalyzedImageDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
            {"video_analyzer_3d/analysed_image/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createSlowVideoAnalyzer2dWideStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SlowVideoAnalyzer2dWideDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(5)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzer2dWideStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer2dWideDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(1)},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy>
    createFastVideoAnalyzer2dWideWithAnalyzedImageStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer2dWideWithAnalyzedImageDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(1)},
            {"video_analyzer_2d_wide/analysed_image/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<AudioAnalyzerDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"audio_analyzer/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createVadStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<VadDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"vad/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createSpeechToTextStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SpeechToTextDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"speech_to_text/filter_state", FilterConfiguration::onOff()},
            {"vad/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}


unique_ptr<BaseStrategy> createExploreStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<ExploreDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{{"explore/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy>
    createFaceAnimationStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle, uint16_t utility)
{
    return make_unique<FaceAnimationStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy>
    createLedEmotionStrategy(shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle, uint16_t utility)
{
    return make_unique<LedEmotionStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createSoundFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SoundFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{{"sound_following/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createNearestFaceFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<NearestFaceFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
            {"nearest_face_following/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createSpecificFaceFollowingStrategy(
    shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<SpecificFaceFollowingStrategy>(utility, move(filterPool), nodeHandle);
}

unique_ptr<BaseStrategy> createSoundObjectPersonFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SoundObjectPersonFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(1)},
            {"sound_object_person_following/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createTalkStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<TalkStrategy>(utility, move(filterPool), move(desireSet), nodeHandle);
}

unique_ptr<BaseStrategy> createGestureStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<GestureStrategy>(utility, move(filterPool), move(desireSet), nodeHandle);
}

unique_ptr<BaseStrategy> createDanceStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<DanceDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}, {"led", 1}},
        unordered_map<string, FilterConfiguration>{
            {"beat_detector/filter_state", FilterConfiguration::onOff()},
            {"head_dance/filter_state", FilterConfiguration::onOff()},
            {"torso_dance/filter_state", FilterConfiguration::onOff()},
            {"led_dance/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}

unique_ptr<BaseStrategy> createPlaySoundStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility)
{
    return make_unique<PlaySoundStrategy>(utility, move(filterPool), move(desireSet), nodeHandle);
}

unique_ptr<BaseStrategy> createTelepresenceStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<TelepresenceDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"sound", 1}},
        unordered_map<string, FilterConfiguration>{{"ego_noise_reduction/filter_state", FilterConfiguration::onOff()}},
        move(filterPool));
}

unique_ptr<BaseStrategy> createTeleoperationStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<TeleoperationDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"motor", 1}},
        unordered_map<string, FilterConfiguration>{
            {"teleoperation/filter_state", FilterConfiguration::onOff()},
        },
        move(filterPool));
}
