#include <t_top_hbba_lite/Strategies.h>

using namespace std;

FaceAnimationStrategy::FaceAnimationStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<rclcpp::Node> node)
    : Strategy<FaceAnimationDesire>(utility, {}, {}, std::move(filterPool)),
      m_node(std::move(node))
{
    m_animationPublisher =
        m_node->create_publisher<std_msgs::msg::String>("face/animation", rclcpp::QoS(1).transient_local());
}

StrategyType FaceAnimationStrategy::strategyType()
{
    return StrategyType::get<FaceAnimationStrategy>();
}

void FaceAnimationStrategy::onEnabling(const FaceAnimationDesire& desire)
{
    std_msgs::msg::String msg;
    msg.data = desire.name();
    m_animationPublisher->publish(msg);
}

void FaceAnimationStrategy::onDisabling()
{
    std_msgs::msg::String msg;
    msg.data = "normal";
    m_animationPublisher->publish(msg);

    Strategy<FaceAnimationDesire>::onDisabling();
}

LedEmotionStrategy::LedEmotionStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<rclcpp::Node> node)
    : Strategy<LedEmotionDesire>(
          utility,
          {},
          {{"led_emotions/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_node(std::move(node))
{
    m_emotionPublisher =
        m_node->create_publisher<std_msgs::msg::String>("led_emotions/name", rclcpp::QoS(1).transient_local());
}

StrategyType LedEmotionStrategy::strategyType()
{
    return StrategyType::get<LedEmotionStrategy>();
}

void LedEmotionStrategy::onEnabling(const LedEmotionDesire& desire)
{
    std_msgs::msg::String msg;
    msg.data = desire.name();
    m_emotionPublisher->publish(msg);
}

LedAnimationStrategy::LedAnimationStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node)
    : Strategy<LedAnimationDesire>(
          utility,
          {},
          {{"led_animations/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_desireSet(desireSet),
      m_node(std::move(node))
{
    m_animationPublisher = m_node->create_publisher<behavior_msgs::msg::LedAnimation>(
        "led_animations/animation",
        rclcpp::QoS(1).transient_local());
    m_animationDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "led_animations/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { animationDoneSubscriberCallback(msg); });
}

StrategyType LedAnimationStrategy::strategyType()
{
    return StrategyType::get<LedAnimationStrategy>();
}

void LedAnimationStrategy::onEnabling(const LedAnimationDesire& desire)
{
    behavior_msgs::msg::LedAnimation msg;
    msg.id = desire.id();
    msg.duration_s = desire.durationS();
    msg.name = desire.name();
    msg.speed = desire.speed();
    msg.colors = desire.colors();
    m_animationPublisher->publish(msg);
}

void LedAnimationStrategy::animationDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    if (msg->id == desireId())
    {
        m_desireSet->removeDesire(msg->id);
    }
}

SpecificFaceFollowingStrategy::SpecificFaceFollowingStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<rclcpp::Node> node)
    : Strategy<SpecificFaceFollowingDesire>(
          utility,
          {},
          {{"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
           {"specific_face_following/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_node(std::move(node))
{
    m_targetNamePublisher =
        m_node->create_publisher<std_msgs::msg::String>("face_following/target_name", rclcpp::QoS(1).transient_local());
}

StrategyType SpecificFaceFollowingStrategy::strategyType()
{
    return StrategyType::get<SpecificFaceFollowingStrategy>();
}

void SpecificFaceFollowingStrategy::onEnabling(const SpecificFaceFollowingDesire& desire)
{
    std_msgs::msg::String msg;
    msg.data = desire.targetName();
    m_targetNamePublisher->publish(msg);
}

TalkStrategy::TalkStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node)
    : Strategy<TalkDesire>(
          utility,
          {{"sound", 1}},
          {{"talk/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_desireSet(std::move(desireSet)),
      m_node(std::move(node))
{
    m_talkPublisher = m_node->create_publisher<behavior_msgs::msg::Text>("talk/text", rclcpp::QoS(1).transient_local());
    m_talkDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "talk/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { talkDoneSubscriberCallback(msg); });
}

StrategyType TalkStrategy::strategyType()
{
    return StrategyType::get<TalkStrategy>();
}

void TalkStrategy::onEnabling(const TalkDesire& desire)
{
    behavior_msgs::msg::Text msg;
    msg.text = desire.text();
    msg.id = desire.id();
    m_talkPublisher->publish(msg);
}

void TalkStrategy::talkDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
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
    shared_ptr<rclcpp::Node> node)
    : Strategy<GestureDesire>(
          utility,
          {},
          {{"gesture/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_desireSet(std::move(desireSet)),
      m_node(std::move(node))
{
    m_gesturePublisher =
        m_node->create_publisher<behavior_msgs::msg::GestureName>("gesture/name", rclcpp::QoS(1).transient_local());
    m_gestureDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "gesture/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { gestureDoneSubscriberCallback(msg); });
}

StrategyType GestureStrategy::strategyType()
{
    return StrategyType::get<GestureStrategy>();
}

void GestureStrategy::onEnabling(const GestureDesire& desire)
{
    behavior_msgs::msg::GestureName msg;
    msg.name = desire.name();
    msg.id = desire.id();
    m_gesturePublisher->publish(msg);
}

void GestureStrategy::gestureDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
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
    shared_ptr<rclcpp::Node> node)
    : Strategy<PlaySoundDesire>(
          utility,
          {{"sound", 1}},
          {{"sound_player/filter_state", FilterConfiguration::onOff()}},
          std::move(filterPool)),
      m_desireSet(desireSet),
      m_node(std::move(node))
{
    m_pathPublisher =
        m_node->create_publisher<behavior_msgs::msg::SoundFile>("sound_player/file", rclcpp::QoS(1).transient_local());
    m_soundDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "sound_player/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { soundDoneSubscriberCallback(msg); });
}

StrategyType PlaySoundStrategy::strategyType()
{
    return StrategyType::get<PlaySoundStrategy>();
}

void PlaySoundStrategy::onEnabling(const PlaySoundDesire& desire)
{
    behavior_msgs::msg::SoundFile msg;
    msg.path = desire.path();
    msg.id = desire.id();
    m_pathPublisher->publish(msg);
}

void PlaySoundStrategy::soundDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    if (msg->id == desireId())
    {
        m_desireSet->removeDesire(msg->id);
    }
}

ChatStrategy::ChatStrategy(
    uint16_t utility,
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node)
    : Strategy<ChatDesire>(
          utility,
          {{"sound", 1}},
          {{"talk/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)},
           {"speech_to_text/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)},
           {"vad/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)},
           {"led_animations/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)},
           {"gesture/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)},
           {"chat/transcript/filter_state", FilterConfiguration::onOff(FilterConfiguration::DefaultState::DISABLED)}},
          std::move(filterPool)),
      m_desireSet(std::move(desireSet)),
      m_node(std::move(node))
{
    m_transcriptSubscriber = m_node->create_subscription<perception_msgs::msg::Transcript>(
        "speech_to_text/transcript",
        1,
        [this](const perception_msgs::msg::Transcript::SharedPtr msg) { transcriptSubscriberCallback(msg); });

    m_transcriptPublisher =
        m_node->create_publisher<perception_msgs::msg::ContextInput>("chat/transcript", rclcpp::QoS(1).transient_local());

    m_chatDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "chat/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { chatDoneSubscriberCallback(msg); });

    m_talkDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "talk/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { talkDoneSubscriberCallback(msg); });

    m_ledAnimationPublisher = m_node->create_publisher<behavior_msgs::msg::LedAnimation>(
        "led_animations/animation",
        rclcpp::QoS(1).transient_local());

    m_ledAnimationDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "led_animations/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { ledAnimationDoneSubscriberCallback(msg); });

    m_gesturePublisher =
        m_node->create_publisher<behavior_msgs::msg::GestureName>("gesture/name", rclcpp::QoS(1).transient_local());

    m_gestureDoneSubscriber = m_node->create_subscription<behavior_msgs::msg::Done>(
        "gesture/done",
        1,
        [this](const behavior_msgs::msg::Done::SharedPtr msg) { gestureDoneSubscriberCallback(msg); });
    
    m_perceptionSubscriberCallback = m_node->create_subscription<perception_msgs::msg::ContextInput>(
        "perception/current_objects",
        1,
        [this](const perception_msgs::msg::ContextInput::SharedPtr msg) { perceptionSubscriberCallback(msg); });

    m_vadTimeoutTimer = m_node->create_wall_timer(std::chrono::seconds(5),
        std::bind(&ChatStrategy::vadTimeoutCallback, this));
}

StrategyType ChatStrategy::strategyType()
{
    return StrategyType::get<ChatStrategy>();
}

void ChatStrategy::onEnabling(const ChatDesire& desire)
{
    // Unused parameter for now
    (void)desire;
    // Start listening
    enableFilter("vad/filter_state");
    enableFilter("speech_to_text/filter_state");

    // Disable chat & talking
    disableFilter("chat/transcript/filter_state");
    disableFilter("talk/filter_state");

    sendListeningLedAnimation();
    isTalking = true;
    m_lastVadTime = std::chrono::steady_clock::now();
}

void ChatStrategy::sendListeningLedAnimation()
{
    enableFilter("led_animations/filter_state");
    behavior_msgs::msg::LedAnimation msg;
    msg.id = desireId().value();
    msg.duration_s = std::numeric_limits<double>::infinity();
    msg.name = "rotating_sin";
    msg.speed = 1.0;
    msg.colors = vector<daemon_ros_client::msg::LedColor>{ChatStrategy::getColor(0, 255, 0)};
    m_ledAnimationPublisher->publish(msg);
}

void ChatStrategy::sendTalkingLedAnimation()
{
    enableFilter("led_animations/filter_state");
    behavior_msgs::msg::LedAnimation msg;
    msg.id = desireId().value();
    msg.duration_s = std::numeric_limits<double>::infinity();
    msg.name = "rotating_sin";
    msg.speed = 1.0;
    msg.colors = vector<daemon_ros_client::msg::LedColor>{ChatStrategy::getColor(255, 0, 0)};
    m_ledAnimationPublisher->publish(msg);
}

void ChatStrategy::transcriptSubscriberCallback(const perception_msgs::msg::Transcript::SharedPtr msg)
{
    if (msg->is_final)
    {
        // Listening done
        disableFilter("vad/filter_state");
        disableFilter("speech_to_text/filter_state");

        // Start chatting
        enableFilter("chat/transcript/filter_state");

        // Start talking
        enableFilter("talk/filter_state");

        sendTalkingLedAnimation();
        //sendGesture("thinking");
        perception_msgs::msg::ContextInput message;
        message.transcript = *msg;
        message.objects = currentObjects;
        message.revive_conversation = false;
        m_transcriptPublisher->publish(message);
        isTalking = true;
    }
}

void ChatStrategy::chatDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    if (msg->ok)
    {
        // Stop chatting
        disableFilter("chat/transcript/filter_state");

        // Stop talking
        disableFilter("talk/filter_state");

        // Start listening
        enableFilter("vad/filter_state");
        enableFilter("speech_to_text/filter_state");

        sendListeningLedAnimation();
        sendGesture("slow_origin_head");

        isTalking = false;
        m_lastVadTime = std::chrono::steady_clock::now();
    }
}

void ChatStrategy::talkDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    static int counter = 0;
    if (msg->ok)
    {
        // Random head position ?
        if (counter++ % 2 == 0)
        {
            //sendGesture("thinking");
        }
        else
        {
            sendGesture("slow_origin_head");
        }
    }
}

void ChatStrategy::ledAnimationDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    if (msg->ok)
    {
        // TODO
    }
}

void ChatStrategy::gestureDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg)
{
    if (msg->id == desireId())
    {
        disableFilter("gesture/filter_state");
    }
}

void ChatStrategy::perceptionSubscriberCallback(const perception_msgs::msg::ContextInput::SharedPtr msg)
{
    currentObjects = msg->objects;
}

void ChatStrategy::vadTimeoutCallback()
{
    if (isTalking){
        m_lastVadTime = std::chrono::steady_clock::now();
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_lastVadTime).count();
    if (elapsed > 30) {
        RCLCPP_WARN(m_node->get_logger(), "Timeout: no activity for 30 seconds.");
        // Listening done
        disableFilter("vad/filter_state");
        disableFilter("speech_to_text/filter_state");

        // Start chatting
        enableFilter("chat/transcript/filter_state");

        // Start talking
        enableFilter("talk/filter_state");

        sendTalkingLedAnimation();
        //sendGesture("thinking");
        perception_msgs::msg::ContextInput message;
        message.transcript = perception_msgs::msg::Transcript();
        message.objects = currentObjects; 
        message.revive_conversation = true;
        m_transcriptPublisher->publish(message);
        isTalking = true;
    }
}

void ChatStrategy::sendGesture(const string& gesture)
{
    enableFilter("gesture/filter_state");
    behavior_msgs::msg::GestureName msg;
    msg.name = gesture;
    msg.id = desireId().value();
    m_gesturePublisher->publish(msg);
}

unique_ptr<BaseStrategy> createCamera3dRecordingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<Camera3dRecordingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_recorder_camera_3d/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createCamera2dWideRecordingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<Camera2dWideRecordingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_recorder_camera_2d_wide/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<RobotNameDetectorDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"robot_name_detector/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy>
    createRobotNameDetectorWithLedStatusDesireStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<RobotNameDetectorWithLedStatusDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"robot_name_detector/filter_state", FilterConfiguration::onOff()},
            {"robot_name_detector/led_status/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createSlowVideoAnalyzer3dStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SlowVideoAnalyzer3dDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(15)},
        },
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzer3dStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer3dDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
        },
        std::move(filterPool));
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
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createSlowVideoAnalyzer2dWideStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SlowVideoAnalyzer2dWideDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(5)},
        },
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createFastVideoAnalyzer2dWideStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<FastVideoAnalyzer2dWideDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(1)},
        },
        std::move(filterPool));
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
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<AudioAnalyzerDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"audio_analyzer/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createVadStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<VadDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"vad/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createSpeechToTextStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SpeechToTextDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"speech_to_text/filter_state", FilterConfiguration::onOff()},
            {"vad/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}


unique_ptr<BaseStrategy> createExploreStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<ExploreDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"explore/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy>
    createFaceAnimationStrategy(shared_ptr<FilterPool> filterPool, shared_ptr<rclcpp::Node> node, uint16_t utility)
{
    return make_unique<FaceAnimationStrategy>(utility, std::move(filterPool), std::move(node));
}

unique_ptr<BaseStrategy>
    createLedEmotionStrategy(shared_ptr<FilterPool> filterPool, shared_ptr<rclcpp::Node> node, uint16_t utility)
{
    return make_unique<LedEmotionStrategy>(utility, std::move(filterPool), std::move(node));
}

unique_ptr<BaseStrategy> createLedAnimationStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<LedAnimationStrategy>(utility, filterPool, desireSet, std::move(node));
}

unique_ptr<BaseStrategy> createSoundFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SoundFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{{"sound_following/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createNearestFaceFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<NearestFaceFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_3d/image_raw/filter_state", FilterConfiguration::throttling(3)},
            {"nearest_face_following/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createSpecificFaceFollowingStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<SpecificFaceFollowingStrategy>(utility, std::move(filterPool), std::move(node));
}

unique_ptr<BaseStrategy> createSoundObjectPersonFollowingStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<SoundObjectPersonFollowingDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"video_analyzer_2d_wide/image_raw/filter_state", FilterConfiguration::throttling(1)},
            {"sound_object_person_following/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createTalkStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<TalkStrategy>(utility, std::move(filterPool), std::move(desireSet), std::move(node));
}

unique_ptr<BaseStrategy> createGestureStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<GestureStrategy>(utility, std::move(filterPool), std::move(desireSet), std::move(node));
}

unique_ptr<BaseStrategy> createDanceStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<DanceDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"beat_detector/filter_state", FilterConfiguration::onOff()},
            {"head_dance/filter_state", FilterConfiguration::onOff()},
            {"torso_dance/filter_state", FilterConfiguration::onOff()},
            {"led_dance/filter_state", FilterConfiguration::onOff()},
        },
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createPlaySoundStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<PlaySoundStrategy>(utility, std::move(filterPool), std::move(desireSet), std::move(node));
}

unique_ptr<BaseStrategy> createTelepresenceStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<TelepresenceDesire>>(
        utility,
        unordered_map<string, uint16_t>{{"sound", 1}},
        unordered_map<string, FilterConfiguration>{{"ego_noise_reduction/filter_state", FilterConfiguration::onOff()}},
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createTeleoperationStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<TeleoperationDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"teleoperation/filter_state", FilterConfiguration::onOff()},
        },
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createTooCloseReactionStrategy(shared_ptr<FilterPool> filterPool, uint16_t utility)
{
    return make_unique<Strategy<TooCloseReactionDesire>>(
        utility,
        unordered_map<string, uint16_t>{},
        unordered_map<string, FilterConfiguration>{
            {"too_close_reaction/filter_state", FilterConfiguration::onOff()},
        },
        std::move(filterPool));
}

unique_ptr<BaseStrategy> createChatStrategy(
    shared_ptr<FilterPool> filterPool,
    shared_ptr<DesireSet> desireSet,
    shared_ptr<rclcpp::Node> node,
    uint16_t utility)
{
    return make_unique<ChatStrategy>(utility, std::move(filterPool), std::move(desireSet), std::move(node));
}
