#ifndef T_TOP_HBBA_LITE_STRATEGIES_H
#define T_TOP_HBBA_LITE_STRATEGIES_H

#include <t_top/hbba_lite/Desires.h>

#include <ros/ros.h>

#include <hbba_lite/core/Strategy.h>

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
};

class TalkStrategy : public Strategy<TalkDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_talkPublisher;

public:
    TalkStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(TalkStrategy);
    DECLARE_NOT_MOVABLE(TalkStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
};

class GestureStrategy : public Strategy<GestureDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_gesturePublisher;

public:
    GestureStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(GestureStrategy);
    DECLARE_NOT_MOVABLE(GestureStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
};

class PlaySoundStrategy : public Strategy<PlaySoundDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_pathPublisher;

public:
    PlaySoundStrategy(uint16_t utility, std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(PlaySoundStrategy);
    DECLARE_NOT_MOVABLE(PlaySoundStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
};


std::unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSlowVideoAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFastVideoAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFastVideoAnalyzerWithAnalyzedImageStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSpeechToTextStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);

std::unique_ptr<BaseStrategy> createExploreStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFaceAnimationStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createSoundFollowingStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createFaceFollowingStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createTalkStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createGestureStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle, uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createDanceStrategy(std::shared_ptr<FilterPool> filterPool,
    uint16_t utility = 1);
std::unique_ptr<BaseStrategy> createPlaySoundStrategy(std::shared_ptr<FilterPool> filterPool,
    ros::NodeHandle& nodeHandle, uint16_t utility = 1);


#endif
