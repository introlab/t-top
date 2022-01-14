#ifndef CONTROL_PANEL_CONTROL_PANEL_STRATEGIES_H
#define CONTROL_PANEL_CONTROL_PANEL_STRATEGIES_H

#include "ControlPanelDesires.h"

#include <ros/ros.h>

#include <hbba_lite/core/Strategy.h>

#include <memory>

class FaceAnimationStrategy : public Strategy<FaceAnimationDesire>
{
    ros::NodeHandle& m_nodeHandle;
    ros::Publisher m_animationPublisher;

public:
    FaceAnimationStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

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
    TalkStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

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
    GestureStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);

    DECLARE_NOT_COPYABLE(GestureStrategy);
    DECLARE_NOT_MOVABLE(GestureStrategy);

protected:
    void onEnabling(const std::unique_ptr<Desire>& desire) override;
};

std::unique_ptr<BaseStrategy> createFaceAnimationStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);
std::unique_ptr<BaseStrategy> createTalkStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);
std::unique_ptr<BaseStrategy> createListenStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createGestureStrategy(std::shared_ptr<FilterPool> filterPool, ros::NodeHandle& nodeHandle);
std::unique_ptr<BaseStrategy> createFaceFollowingStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createSoundFollowingStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createDanceStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createExploreStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createVideoAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createAudioAnalyzerStrategy(std::shared_ptr<FilterPool> filterPool);
std::unique_ptr<BaseStrategy> createRobotNameDetectorStrategy(std::shared_ptr<FilterPool> filterPool);

#endif
