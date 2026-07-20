#ifndef T_TOP_HBBA_LITE_STRATEGIES_H
#define T_TOP_HBBA_LITE_STRATEGIES_H

#include <t_top_hbba_lite/Desires.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>

#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>

#include <behavior_msgs/msg/led_animation.hpp>
#include <behavior_msgs/msg/text.hpp>
#include <behavior_msgs/msg/gesture_name.hpp>
#include <behavior_msgs/msg/done.hpp>
#include <behavior_msgs/msg/sound_file.hpp>

#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/filters/FilterState.h>
#include <optional>

#include <perception_msgs/msg/transcript.hpp>
#include <perception_msgs/msg/context_input.hpp>


#include <memory>

class FaceAnimationStrategy : public Strategy<FaceAnimationDesire>
{
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_animationPublisher;

public:
    FaceAnimationStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(FaceAnimationStrategy);
    DECLARE_NOT_MOVABLE(FaceAnimationStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const FaceAnimationDesire& desire) override;
    void onDisabling() override;
};

class LedEmotionStrategy : public Strategy<LedEmotionDesire>
{
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_emotionPublisher;

public:
    LedEmotionStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(LedEmotionStrategy);
    DECLARE_NOT_MOVABLE(LedEmotionStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const LedEmotionDesire& desire) override;
};

class LedAnimationStrategy : public Strategy<LedAnimationDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<behavior_msgs::msg::LedAnimation>::SharedPtr m_animationPublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_animationDoneSubscriber;

public:
    LedAnimationStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(LedAnimationStrategy);
    DECLARE_NOT_MOVABLE(LedAnimationStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const LedAnimationDesire& desire) override;

private:
    void animationDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
};

class SpecificFaceFollowingStrategy : public Strategy<SpecificFaceFollowingDesire>
{
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_targetNamePublisher;

public:
    SpecificFaceFollowingStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(SpecificFaceFollowingStrategy);
    DECLARE_NOT_MOVABLE(SpecificFaceFollowingStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const SpecificFaceFollowingDesire& desire) override;
};

class TalkStrategy : public Strategy<TalkDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<behavior_msgs::msg::Text>::SharedPtr m_talkPublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_talkDoneSubscriber;

public:
    TalkStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(TalkStrategy);
    DECLARE_NOT_MOVABLE(TalkStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const TalkDesire& desire) override;

private:
    void talkDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
};

class GestureStrategy : public Strategy<GestureDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<behavior_msgs::msg::GestureName>::SharedPtr m_gesturePublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_gestureDoneSubscriber;

public:
    GestureStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(GestureStrategy);
    DECLARE_NOT_MOVABLE(GestureStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const GestureDesire& desire) override;

private:
    void gestureDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
};

class PlaySoundStrategy : public Strategy<PlaySoundDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<rclcpp::Node> m_node;
    rclcpp::Publisher<behavior_msgs::msg::SoundFile>::SharedPtr m_pathPublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_soundDoneSubscriber;

public:
    PlaySoundStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        std::shared_ptr<rclcpp::Node> node);

    DECLARE_NOT_COPYABLE(PlaySoundStrategy);
    DECLARE_NOT_MOVABLE(PlaySoundStrategy);

    StrategyType strategyType() override;

protected:
    void onEnabling(const PlaySoundDesire& desire) override;

private:
    void soundDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
};

class ChatStrategy : public Strategy<ChatDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    std::shared_ptr<rclcpp::Node> m_node;

    rclcpp::Subscription<perception_msgs::msg::Transcript>::SharedPtr m_transcriptSubscriber;
    rclcpp::Publisher<perception_msgs::msg::ContextInput>::SharedPtr m_contextInputPublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_chatDoneSubscriber;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_talkDoneSubscriber;

    // LEDS
    rclcpp::Publisher<behavior_msgs::msg::LedAnimation>::SharedPtr m_ledAnimationPublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_ledAnimationDoneSubscriber;

    // GESTURES
    rclcpp::Publisher<behavior_msgs::msg::GestureName>::SharedPtr m_gesturePublisher;
    rclcpp::Subscription<behavior_msgs::msg::Done>::SharedPtr m_gestureDoneSubscriber;

public:
    ChatStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        std::shared_ptr<rclcpp::Node> node,
        std::shared_ptr<const std::unordered_set<std::string>> removedFeatureSet);

    DECLARE_NOT_COPYABLE(ChatStrategy);
    DECLARE_NOT_MOVABLE(ChatStrategy);

    StrategyType strategyType() override;

    static daemon_ros_client::msg::LedColor getColor(uint8_t r, uint8_t g, uint8_t b)
    {
        daemon_ros_client::msg::LedColor c;
        c.red = r;
        c.green = g;
        c.blue = b;
        return c;
    }

protected:
    void onEnabling(const ChatDesire& desire) override;

private:
    void transcriptSubscriberCallback(const perception_msgs::msg::Transcript::SharedPtr msg);
    void chatDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
    void talkDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
    void ledAnimationDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);
    void gestureDoneSubscriberCallback(const behavior_msgs::msg::Done::SharedPtr msg);

    void sendListeningLedAnimation();
    void sendTalkingLedAnimation();
    void sendGesture(const std::string& gesture);

    std::vector<std::string> currentObjects;
    std::shared_ptr<const std::unordered_set<std::string> > m_removedFeatureSet;

};


std::unique_ptr<BaseStrategy>
    createCamera3dRecordingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createCamera2dWideRecordingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createRobotNameDetectorStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createRobotNameDetectorWithLedStatusDesireStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSlowVideoAnalyzer3dStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createFastVideoAnalyzer3dStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createFastVideoAnalyzer3dWithAnalyzedImageStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSlowVideoAnalyzer2dWideStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createFastVideoAnalyzer2dWideStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFastVideoAnalyzer2dWideWithAnalyzedImageStrategy(
    std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createVadStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSpeechToTextStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createEouStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createExploreStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFaceAnimationStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createLedEmotionStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createLedAnimationStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSoundFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createNearestFaceFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSpecificFaceFollowingStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSoundObjectPersonFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTalkStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createGestureStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createDanceStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createPlaySoundStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    std::shared_ptr<rclcpp::Node> node,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTelepresenceStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTeleoperationStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createTooCloseReactionStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createChatStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<const std::unordered_set<std::string>> removedFeatureSet,
    uint16_t utility = 1);

#endif
