#ifndef T_TOP_HBBA_LITE_STRATEGIES_H
#define T_TOP_HBBA_LITE_STRATEGIES_H

#include <t_top_hbba_lite/Desires.h>

#include <ros/ros.h>

#include <hbba_lite/core/Strategy.h>

#include <talk/Done.h>
#include <gesture/Done.h>
#include <sound_player/Done.h>

#include <memory>

class FaceAnimationStrategy : public Strategy<FaceAnimationDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_animationPublisher;

public:
    FaceAnimationStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(FaceAnimationStrategy);
    DECLARE_NOT_MOVABLE(FaceAnimationStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
    void onDisabling() override;
};

class SpecificFaceFollowingStrategy : public Strategy<SpecificFaceFollowingDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_targetNamePublisher;

public:
    SpecificFaceFollowingStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(SpecificFaceFollowingStrategy);
    DECLARE_NOT_MOVABLE(SpecificFaceFollowingStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
};

class TalkStrategy : public Strategy<TalkDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_talkPublisher;
    ros::Subscriber m_talkDoneSubscriber;

public:
    TalkStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(TalkStrategy);
    DECLARE_NOT_MOVABLE(TalkStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;

private:
    void talkDoneSubscriberCallback(const talk::Done::ConstPtr& msg);
};

class GestureStrategy : public Strategy<GestureDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_gesturePublisher;
    ros::Subscriber m_gestureDoneSubscriber;

public:
    GestureStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(GestureStrategy);
    DECLARE_NOT_MOVABLE(GestureStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;

private:
    void gestureDoneSubscriberCallback(const gesture::Done::ConstPtr& msg);
};

class PlaySoundStrategy : public Strategy<PlaySoundDesire>
{
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_pathPublisher;
    ros::Subscriber m_soundDoneSubscriber;

public:
    PlaySoundStrategy(
        uint16_t utility,
        std::shared_ptr<FilterPool> filterPool,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(PlaySoundStrategy);
    DECLARE_NOT_MOVABLE(PlaySoundStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;

private:
    void soundDoneSubscriberCallback(const sound_player::Done::ConstPtr& msg);
};

std::unique_ptr<BaseStrategy>
    createCamera3dRecordingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createCamera2dWideRecordingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);

std::unique_ptr<BaseStrategy>
    createRobotNameDetectorStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
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
std::unique_ptr<BaseStrategy> createSpeechToTextStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);

std::unique_ptr<BaseStrategy> createExploreStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFaceAnimationStrategy(
    std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSoundFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createNearestFaceFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSpecificFaceFollowingStrategy(
    std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy>
    createSoundObjectPersonFollowingStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTalkStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createGestureStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createDanceStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createPlaySoundStrategy(
    std::shared_ptr<FilterPool> filterPool,
    std::shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    uint16_t utility = 1);

std::unique_ptr<BaseStrategy> createTelepresenceStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTeleoperationStrategy(std::shared_ptr<FilterPool> filterPool, uint16_t utility = 1);


#endif
