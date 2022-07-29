#include <ros/ros.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/core/RosFilterPool.h>
#include <hbba_lite/core/GecodeSolver.h>
#include <hbba_lite/core/HbbaLite.h>

#include <t_top_hbba_lite/Strategies.h>

#include <memory>

using namespace std;

constexpr bool WAIT_FOR_SERVICE = true;

void startNode(
    ros::NodeHandle& nodeHandle,
    bool camera2dWideEnabled,
    bool recordSession)
{
    auto desireSet = make_shared<DesireSet>();
    auto filterPool = make_shared<RosFilterPool>(nodeHandle, WAIT_FOR_SERVICE);

    vector<unique_ptr<BaseStrategy>> strategies;
    if (recordSession)
    {
        strategies.emplace_back(createCamera3dRecordingStrategy(filterPool));
    }
    if (recordSession && camera2dWideEnabled)
    {
        strategies.emplace_back(createCamera2dWideRecordingStrategy(filterPool));
    }

    strategies.emplace_back(createRobotNameDetectorStrategy(filterPool));
    strategies.emplace_back(createFastVideoAnalyzer3dStrategy(filterPool));
    strategies.emplace_back(createAudioAnalyzerStrategy(filterPool));
    strategies.emplace_back(createSpeechToTextStrategy(filterPool));

    strategies.emplace_back(createFaceAnimationStrategy(filterPool, nodeHandle));
    strategies.emplace_back(createNearestFaceFollowingStrategy(filterPool));
    strategies.emplace_back(createSoundFollowingStrategy(filterPool));
    strategies.emplace_back(createTalkStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createGestureStrategy(filterPool, desireSet, nodeHandle));
    strategies.emplace_back(createPlaySoundStrategy(filterPool, desireSet, nodeHandle));

    ros::spin();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "home_logger_node");
    ros::NodeHandle nodeHandle;
    ros::NodeHandle privateNodeHandle("~");

    bool camera2dWideEnabled;
    if (!privateNodeHandle.getParam("camera_2d_wide_enabled", camera2dWideEnabled))
    {
        ROS_ERROR("The parameter camera_2d_wide_enabled must be set.");
        return -1;
    }

    bool recordSession;
    if (!privateNodeHandle.getParam("record_session", recordSession))
    {
        ROS_ERROR("The parameter record_session must be set.");
        return -1;
    }

    startNode(nodeHandle, camera2dWideEnabled, recordSession);

    return 0;
}
